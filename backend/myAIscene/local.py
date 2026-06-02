"""LocalRenderer — the real HuggingFace implementation of the Renderer
protocol.

Trust separation (CONSTITUTION Article II): every model that produces
content is separated from the model that verifies it.
- Kokoro speaks; Whisper transcribes for narration_verify.
- SDXL-turbo generates stills; CLIP scores them for footage_verify.
- MusicGen generates beds; duration math (not a model) gates them.
"""
from __future__ import annotations

import os
import subprocess
import wave
from pathlib import Path
from typing import Protocol

from .renderers import ClipOut, MusicOut, NarrationOut, ProbeOut, StillOut
from .spec import Beat, ProductionSpec


# ---- environment setup --------------------------------------------------

def _ensure_espeak_env() -> None:
    lib = "/opt/homebrew/lib/libespeak-ng.dylib"
    data = "/opt/homebrew/share/espeak-ng-data"
    if os.path.exists(lib):
        os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", lib)
    if os.path.isdir(data):
        os.environ.setdefault("ESPEAK_DATA_PATH", data)


# ---- backend protocols --------------------------------------------------

class TTSEngine(Protocol):
    def synth(self, text: str, voice: str) -> tuple[object, int]: ...


class ASREngine(Protocol):
    def transcribe(self, wav_path: str) -> str: ...


class T2IEngine(Protocol):
    def generate(self, prompt: str, path: Path) -> None: ...


class CLIPEngine(Protocol):
    def score(self, image_path: str, text: str) -> float: ...


class MusicEngine(Protocol):
    def generate(self, prompt: str, duration_s: float, path: Path) -> None: ...


# ---- real backends (lazy imports) ---------------------------------------

class KokoroTTS:
    SAMPLE_RATE = 24000

    def __init__(self, lang_code="b", default_voice="bm_george", speed=1.0):
        self.lang_code = lang_code
        self.default_voice = default_voice
        self.speed = speed
        self._pipeline = None

    def _ensure(self):
        if self._pipeline is None:
            _ensure_espeak_env()
            from kokoro import KPipeline
            self._pipeline = KPipeline(lang_code=self.lang_code)
        return self._pipeline

    def synth(self, text: str, voice: str | None = None):
        import numpy as np
        pipe = self._ensure()
        chunks = []
        for _g, _p, audio in pipe(text, voice=voice or self.default_voice, speed=self.speed):
            arr = audio.detach().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio)
            chunks.append(arr.astype("float32"))
        if not chunks:
            return np.zeros(0, dtype="float32"), self.SAMPLE_RATE
        return np.concatenate(chunks), self.SAMPLE_RATE


class WhisperASR:
    def __init__(self, model_size="base", device="cpu", compute_type="int8"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def _ensure(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
        return self._model

    def transcribe(self, wav_path: str) -> str:
        m = self._ensure()
        segs, _ = m.transcribe(wav_path, language="en")
        return " ".join(s.text.strip() for s in segs).strip()


class SDXLTurboT2I:
    """stabilityai/sdxl-turbo, 4-step fp16 MPS. License: non-commercial."""

    def __init__(self, model_id="stabilityai/sdxl-turbo", gen_size=(1024, 576), steps=4):
        self.model_id = model_id
        self.gen_size = gen_size
        self.steps = steps
        self._pipe = None

    def _ensure(self):
        if self._pipe is None:
            import torch
            from diffusers import AutoPipelineForText2Image
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            dtype = torch.float16 if device == "mps" else torch.float32
            self._pipe = AutoPipelineForText2Image.from_pretrained(
                self.model_id, torch_dtype=dtype,
                variant="fp16" if dtype == torch.float16 else None,
            ).to(device)
        return self._pipe

    def generate(self, prompt: str, path: Path) -> None:
        w, h = self.gen_size
        image = self._ensure()(prompt=prompt, num_inference_steps=self.steps,
                               guidance_scale=0.0, width=w, height=h).images[0]
        image.save(str(path))


class TransformersCLIPScorer:
    """openai/clip-vit-large-patch14 — cosine similarity for footage_verify."""

    def __init__(self, model_id="openai/clip-vit-large-patch14"):
        self.model_id = model_id
        self._model = None
        self._proc = None

    def _ensure(self):
        if self._model is None:
            from transformers import CLIPModel, CLIPProcessor
            self._proc = CLIPProcessor.from_pretrained(self.model_id)
            self._model = CLIPModel.from_pretrained(self.model_id)
            self._model.eval()
        return self._model, self._proc

    def score(self, image_path: str, text: str) -> float:
        import torch
        from PIL import Image
        model, proc = self._ensure()
        img = Image.open(image_path).convert("RGB")
        inputs = proc(text=[text], images=[img], return_tensors="pt",
                      padding=True, truncation=True)
        with torch.no_grad():
            out = model(**inputs)
        ie = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        te = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
        return float((ie * te).sum().clamp(0.0, 1.0))


class MusicGenEngine:
    """facebook/musicgen-small (300M, ~fast on MPS) for per-beat music beds.
    Upgrade to musicgen-medium via model_id kwarg for better quality.
    License: CC-BY-NC — portfolio use only.
    """
    SAMPLE_RATE = 32000
    TOKENS_PER_SEC = 50

    def __init__(self, model_id="facebook/musicgen-small"):
        self.model_id = model_id
        self._model = None
        self._proc = None

    def _ensure(self):
        if self._model is None:
            import torch
            from transformers import AutoProcessor, MusicgenForConditionalGeneration
            self._proc = AutoProcessor.from_pretrained(self.model_id)
            self._model = MusicgenForConditionalGeneration.from_pretrained(self.model_id)
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            try:
                self._model = self._model.to(device)
            except Exception:
                pass  # stay on CPU if MPS op unsupported
        return self._model, self._proc

    def generate(self, prompt: str, duration_s: float, path: Path) -> None:
        import numpy as np
        import torch
        model, proc = self._ensure()
        max_tokens = int(duration_s * self.TOKENS_PER_SEC) + 50
        inputs = proc(text=[prompt], padding=True, return_tensors="pt")
        # move inputs to model device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            audio = model.generate(**inputs, max_new_tokens=max_tokens)
        samples = audio[0, 0].cpu().numpy().astype("float32")
        write_wav(path, samples, self.SAMPLE_RATE)


# ---- helpers ------------------------------------------------------------

def write_wav(path: str | Path, samples, sample_rate: int) -> None:
    import numpy as np
    pcm = np.clip(np.asarray(samples, dtype="float32"), -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())


def _ffprobe_facts(path: str | Path) -> ProbeOut:
    p = str(path)
    if not os.path.exists(p):
        return ProbeOut(path=p, exists=False, duration_s=0.0,
                        has_audio=False, resolution=(0, 0))
    dur = float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", p],
        capture_output=True, text=True, check=True,
    ).stdout.strip())
    res_out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", p],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    w, h = (int(x) for x in res_out.split(",")) if res_out else (0, 0)
    has_audio = bool(subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=codec_type",
         "-of", "default=noprint_wrappers=1:nokey=1", p],
        capture_output=True, text=True,
    ).stdout.strip())
    return ProbeOut(path=p, exists=True, duration_s=dur, has_audio=has_audio, resolution=(w, h))


def _ffprobe_duration(path: str | Path) -> float:
    return float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip())


_PAN_PRESETS = [
    lambda W, H, pw, ph, d: f"scale={pw}:{ph},crop={W}:{H}:x='min(({pw}-{W})*t/{d},{pw}-{W})':y='({ph}-{H})/2',format=yuv420p",
    lambda W, H, pw, ph, d: f"scale={pw}:{ph},crop={W}:{H}:x='({pw}-{W})/2':y='min(({ph}-{H})*t/{d},{ph}-{H})',format=yuv420p",
    lambda W, H, pw, ph, d: f"scale={pw}:{ph},crop={W}:{H}:x='max(({pw}-{W})*(1-t/{d}),0)':y='({ph}-{H})/2',format=yuv420p",
    lambda W, H, pw, ph, d: f"scale={pw}:{ph},crop={W}:{H}:x='({pw}-{W})/2':y='max(({ph}-{H})*(1-t/{d}),0)',format=yuv420p",
]


def _ken_burns_filter(beat_idx: int, W: int, H: int, duration_s: float) -> str:
    pad_w, pad_h = int(W * 1.1), int(H * 1.1)
    return _PAN_PRESETS[beat_idx % len(_PAN_PRESETS)](W, H, pad_w, pad_h, duration_s)


def _find_system_font(size: int) -> "ImageFont":
    """Return a PIL ImageFont at `size` pt. Tries system serif fonts; falls back
    to PIL's built-in bitmap font (always available)."""
    from PIL import ImageFont
    for p in [
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        "/Library/Fonts/Times New Roman.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ]:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _music_prompt(beat: Beat) -> str | None:
    if not beat.music:
        return None
    parts = list(beat.music.get("keywords", []))
    mood = beat.music.get("mood", "")
    if mood and mood not in parts:
        parts.append(mood)
    return ", ".join(parts) if parts else "cinematic ambient documentary background music"


# ---- the renderer -------------------------------------------------------

class LocalRenderer:
    def __init__(
        self,
        out_dir: str | Path = "/tmp/myAIscene",
        tts: TTSEngine | None = None,
        asr: ASREngine | None = None,
        voice: str | None = None,
        t2i: T2IEngine | None = None,
        clip: CLIPEngine | None = None,
        music_engine: MusicEngine | None = None,
        fps: int = 24,
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.tts = tts if tts is not None else KokoroTTS()
        self.asr = asr if asr is not None else WhisperASR()
        self.voice = voice
        self.t2i = t2i if t2i is not None else SDXLTurboT2I()
        self.clip = clip if clip is not None else TransformersCLIPScorer()
        self.music_engine = music_engine if music_engine is not None else MusicGenEngine()
        self.fps = fps

    # ---- Phase 2: narration ---

    def narrate(self, beat: Beat, spec: ProductionSpec) -> NarrationOut:
        voice = self.voice or spec.episode.voice.voice
        samples, sr = self.tts.synth(beat.narration, voice=voice)
        path = self.out_dir / f"{beat.id}.wav"
        write_wav(path, samples, sr)
        duration_s = (len(samples) / sr) if sr else 0.0
        transcript = self.asr.transcribe(str(path))
        return NarrationOut(audio_path=str(path), transcript=transcript, duration_s=duration_s)

    # ---- Phase 3: visuals ---

    def still(self, beat: Beat, spec: ProductionSpec) -> StillOut:
        path = self.out_dir / f"{beat.id}.png"
        self.t2i.generate(beat.footage_prompt, path)
        score = self.clip.score(str(path), beat.footage_prompt)
        return StillOut(image_path=str(path), clip_score=score)

    def motion(self, beat: Beat, still: StillOut, spec: ProductionSpec) -> ClipOut:
        W, H = spec.episode.resolution
        beat_idx = next((i for i, b in enumerate(spec.beats) if b.id == beat.id), 0)
        vf = _ken_burns_filter(beat_idx, W, H, beat.duration_s)
        out_path = self.out_dir / f"{beat.id}_clip.mp4"
        subprocess.run(
            ["ffmpeg", "-loop", "1", "-i", str(still.image_path),
             "-vf", vf, "-t", str(beat.duration_s), "-r", str(self.fps),
             "-c:v", "libx264", "-preset", "ultrafast", "-tune", "stillimage",
             "-y", str(out_path)],
            check=True, capture_output=True,
        )
        return ClipOut(clip_path=str(out_path), duration_s=_ffprobe_duration(out_path))

    # ---- Phase 4: music ---

    def music(self, beat: Beat, spec: ProductionSpec) -> MusicOut | None:
        prompt = _music_prompt(beat)
        if prompt is None:
            return None
        path = self.out_dir / f"{beat.id}_music.wav"
        self.music_engine.generate(prompt, beat.duration_s, path)
        dur = _ffprobe_duration(path)
        return MusicOut(asset_path=str(path), duration_s=dur)

    # ---- Phase 5: grade + assemble ---

    def assemble(
        self,
        beats: list,
        beat_results: list,
        spec: ProductionSpec,
        out_path: str = "/tmp/myAIscene/episode.mp4",
        *,
        emitter=None,
    ) -> ProbeOut:
        from .luts import ensure_luts, lut_for

        def _emit(event, stage, **kw):
            if emitter:
                emitter.emit(event, stage, **kw)

        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        W, H = spec.episode.resolution
        music_vol = 10 ** (spec.audio.vo_db / 20)  # -6dB → ~0.501 linear

        # 1. Generate LUT files
        lut_paths = ensure_luts(self.out_dir / "luts")

        # 2. Per-beat: grade video + mix VO/music → {id}_av.mp4
        av_paths: list[str] = []
        total = len(beats)
        for i, (beat, br) in enumerate(zip(beats, beat_results)):
            _emit("step_start", "grade_mix", beat=beat.id, index=i+1, total=total)
            lut_file = str(lut_for(beat.grade.get("lut", ""), lut_paths))
            clip_path = br.clip.clip_path if br.clip else None
            av_path = str(self.out_dir / f"{beat.id}_av.mp4")

            if clip_path and os.path.exists(clip_path):
                self._grade_and_mix(
                    clip_path=clip_path,
                    narr_path=br.narr_path,
                    music_path=br.music_path if not br.music_dropped else None,
                    lut_file=lut_file,
                    window_s=beat.duration_s,
                    music_vol=music_vol,
                    out_path=av_path,
                )
            else:
                subprocess.run([
                    "ffmpeg", "-f", "lavfi",
                    "-i", f"color=c=black:s={W}x{H}:r={self.fps}",
                    "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
                    "-t", str(beat.duration_s),
                    "-c:v", "libx264", "-preset", "ultrafast",
                    "-c:a", "aac", "-y", av_path,
                ], check=True, capture_output=True)
            av_paths.append(av_path)
            _emit("step_complete", "grade_mix", beat=beat.id, index=i+1, total=total)

        # 3. Concatenate beats
        _emit("step_start", "concat", clips=total)
        main_path = str(self.out_dir / "episode_main.mp4")
        concat_list = self.out_dir / "concat.txt"
        abs_paths = [os.path.abspath(p) for p in av_paths]
        concat_list.write_text("\n".join(f"file '{p}'" for p in abs_paths) + "\n")
        subprocess.run([
            "ffmpeg", "-f", "concat", "-safe", "0", "-i", str(concat_list),
            "-c", "copy", "-y", main_path,
        ], check=True, capture_output=True)
        _emit("step_complete", "concat")

        # 4. Title card
        tc = spec.titlecard
        if tc:
            _emit("step_start", "titlecard")
            titled_path = str(self.out_dir / "episode_titled.mp4")
            self._prepend_title(main_path, tc, spec.episode, W, H, titled_path)
            _emit("step_complete", "titlecard")
        else:
            titled_path = main_path

        # 5. Film grain
        _emit("step_start", "grain")
        grain = max(1, int(spec.episode.grain * 100))
        subprocess.run([
            "ffmpeg", "-i", titled_path,
            "-vf", f"noise=alls={grain}:allf=t+u",
            "-c:a", "copy", "-c:v", "libx264", "-preset", "medium",
            "-y", str(out),
        ], check=True, capture_output=True)
        _emit("step_complete", "grain", path=str(out))

        return _ffprobe_facts(str(out))

    def _grade_and_mix(
        self, clip_path: str, narr_path: str, music_path: str | None,
        lut_file: str, window_s: float, music_vol: float, out_path: str,
    ) -> None:
        vf = f"lut3d=file={lut_file}"
        if music_path and os.path.exists(music_path):
            filt = (
                f"[0:v]{vf}[gv];"
                f"[1:a]apad=whole_dur={window_s}[vo];"
                f"[2:a]volume={music_vol:.4f}[mu];"
                f"[vo][mu]amix=inputs=2:normalize=0:duration=first[mix];"
                f"[mix]apad=whole_dur={window_s}[out_a]"
            )
            subprocess.run([
                "ffmpeg", "-i", clip_path, "-i", narr_path, "-i", music_path,
                "-filter_complex", filt,
                "-map", "[gv]", "-map", "[out_a]",
                "-t", str(window_s), "-r", str(self.fps),
                "-c:v", "libx264", "-preset", "ultrafast",
                "-c:a", "aac", "-ar", "44100",
                "-y", out_path,
            ], check=True, capture_output=True)
        else:
            filt = f"[0:v]{vf}[gv];[1:a]apad=whole_dur={window_s}[out_a]"
            subprocess.run([
                "ffmpeg", "-i", clip_path, "-i", narr_path,
                "-filter_complex", filt,
                "-map", "[gv]", "-map", "[out_a]",
                "-t", str(window_s), "-r", str(self.fps),
                "-c:v", "libx264", "-preset", "ultrafast",
                "-c:a", "aac", "-ar", "44100",
                "-y", out_path,
            ], check=True, capture_output=True)

    def _prepend_title(
        self, main_path: str, tc: dict, ep, W: int, H: int, out_path: str,
    ) -> None:
        """Generate a title card using PIL (avoids libfreetype dep on ffmpeg),
        fade it in with ffmpeg's built-in fade filter, and prepend it."""
        import numpy as np
        from PIL import Image, ImageDraw

        fade_s = float(tc.get("fade_s", 2.0))
        tc_dur = fade_s + 0.5
        title_text = tc.get("text", ep.title)
        subtitle_text = tc.get("subtitle", "")

        # Draw the title card frame with PIL
        img = Image.new("RGB", (W, H), color=(0, 0, 0))
        draw = ImageDraw.Draw(img)
        font_large = _find_system_font(72)
        font_small = _find_system_font(36)
        draw.text((W // 2, H // 2 - 50), title_text, fill=(255, 255, 255),
                  font=font_large, anchor="mm")
        draw.text((W // 2, H // 2 + 50), subtitle_text, fill=(200, 200, 200),
                  font=font_small, anchor="mm")

        tc_png = str(self.out_dir / "titlecard.png")
        tc_video = str(self.out_dir / "titlecard_v.mp4")
        tc_audio = str(self.out_dir / "titlecard_a.wav")
        tc_av = str(self.out_dir / "titlecard_av.mp4")

        img.save(tc_png)

        # Title card video: static PIL image + fade filter (no drawtext/libfreetype needed)
        subprocess.run([
            "ffmpeg", "-loop", "1", "-i", tc_png,
            "-vf", f"fade=t=in:st=0:d={fade_s},format=yuv420p",
            "-t", str(tc_dur), "-r", str(self.fps),
            "-c:v", "libx264", "-preset", "ultrafast", "-y", tc_video,
        ], check=True, capture_output=True)

        write_wav(tc_audio, np.zeros(int(44100 * tc_dur), dtype="float32"), 44100)

        subprocess.run([
            "ffmpeg", "-i", tc_video, "-i", tc_audio,
            "-c:v", "copy", "-c:a", "aac", "-t", str(tc_dur), "-y", tc_av,
        ], check=True, capture_output=True)

        prepend = self.out_dir / "prepend.txt"
        prepend.write_text(
            f"file '{os.path.abspath(tc_av)}'\nfile '{os.path.abspath(main_path)}'\n"
        )
        subprocess.run([
            "ffmpeg", "-f", "concat", "-safe", "0", "-i", str(prepend),
            "-c", "copy", "-y", out_path,
        ], check=True, capture_output=True)
