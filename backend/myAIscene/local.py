"""LocalRenderer — the real HuggingFace implementation of the Renderer
protocol.

Trust separation (CONSTITUTION Article II): every model that produces
content is separated from the model that verifies it. Kokoro speaks;
Whisper transcribes for narration_verify. SDXL-turbo generates stills;
CLIP scores them for footage_verify. The thing being graded never grades
itself — the same judge-separation discipline as my-AI-stro's Mistral
judge.
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
    """Point Kokoro's British G2P at the Homebrew espeak-ng dylib.
    No-op when already set or dylib absent. Pre-install en_core_web_sm
    (`python -m spacy download en_core_web_sm`) to avoid a first-synth
    network stall from misaki's lazy pip install.
    """
    lib = "/opt/homebrew/lib/libespeak-ng.dylib"
    data = "/opt/homebrew/share/espeak-ng-data"
    if os.path.exists(lib):
        os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", lib)
    if os.path.isdir(data):
        os.environ.setdefault("ESPEAK_DATA_PATH", data)


# ---- backend protocols (injected fakes drive tests without downloads) ---

class TTSEngine(Protocol):
    def synth(self, text: str, voice: str) -> tuple[object, int]: ...


class ASREngine(Protocol):
    def transcribe(self, wav_path: str) -> str: ...


class T2IEngine(Protocol):
    """Text-to-image: prompt → saves PNG at `path`, returns PIL Image."""
    def generate(self, prompt: str, path: Path) -> None: ...


class CLIPEngine(Protocol):
    """Cosine similarity between an image file and a text prompt in [0,1]."""
    def score(self, image_path: str, text: str) -> float: ...


# ---- real backends (lazy imports: nothing heavy until first use) --------

class KokoroTTS:
    """hexgrad/Kokoro-82M — British male `bm_george`, 24 kHz mono."""
    SAMPLE_RATE = 24000

    def __init__(self, lang_code: str = "b", default_voice: str = "bm_george", speed: float = 1.0):
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
    """faster-whisper base, CPU int8 — accurate enough for the word-match gate."""
    def __init__(self, model_size: str = "base", device: str = "cpu", compute_type: str = "int8"):
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
    """stabilityai/sdxl-turbo — 4-step, fp16, MPS. Generates at gen_size
    (default 1024×576, 16:9). The motion() stage scales to the episode
    resolution, so gen_size is intentionally smaller than final output.
    License: non-commercial (Stability NC) — portfolio use only."""

    def __init__(
        self,
        model_id: str = "stabilityai/sdxl-turbo",
        gen_size: tuple[int, int] = (1024, 576),
        steps: int = 4,
    ):
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
                self.model_id, torch_dtype=dtype, variant="fp16" if dtype == torch.float16 else None,
            ).to(device)
        return self._pipe

    def generate(self, prompt: str, path: Path) -> None:
        w, h = self.gen_size
        pipe = self._ensure()
        image = pipe(prompt=prompt, num_inference_steps=self.steps,
                     guidance_scale=0.0, width=w, height=h).images[0]
        image.save(str(path))


class TransformersCLIPScorer:
    """openai/clip-vit-large-patch14 via transformers. Returns cosine
    similarity in ~[0, 1]. The footage_verify gate threshold is 0.22
    (ARCHITECTURE.md) — SDXL-turbo with a good prompt easily exceeds that."""

    def __init__(self, model_id: str = "openai/clip-vit-large-patch14"):
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
        inputs = proc(text=[text], images=[img], return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            out = model(**inputs)
        ie = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        te = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
        return float((ie * te).sum().clamp(0.0, 1.0))


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


def _ffprobe_duration(path: str | Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(r.stdout.strip())


# 4 pan presets cycling by beat index (mod 4). Each scales the PNG to
# pad_w×pad_h (10% larger than target), then crops to W×H with a moving
# offset so the image drifts slowly — the classic Ken Burns effect.
_PAN_PRESETS = [
    # pan left-to-right
    lambda W, H, pw, ph, d: f"scale={pw}:{ph},crop={W}:{H}:x='min(({pw}-{W})*t/{d},{pw}-{W})':y='({ph}-{H})/2',format=yuv420p",
    # pan top-to-bottom
    lambda W, H, pw, ph, d: f"scale={pw}:{ph},crop={W}:{H}:x='({pw}-{W})/2':y='min(({ph}-{H})*t/{d},{ph}-{H})',format=yuv420p",
    # pan right-to-left
    lambda W, H, pw, ph, d: f"scale={pw}:{ph},crop={W}:{H}:x='max(({pw}-{W})*(1-t/{d}),0)':y='({ph}-{H})/2',format=yuv420p",
    # pan bottom-to-top
    lambda W, H, pw, ph, d: f"scale={pw}:{ph},crop={W}:{H}:x='({pw}-{W})/2':y='max(({ph}-{H})*(1-t/{d}),0)',format=yuv420p",
]


def _ken_burns_filter(beat_idx: int, W: int, H: int, duration_s: float) -> str:
    pad_w = int(W * 1.1)
    pad_h = int(H * 1.1)
    return _PAN_PRESETS[beat_idx % len(_PAN_PRESETS)](W, H, pad_w, pad_h, duration_s)


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
        fps: int = 24,
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.tts = tts if tts is not None else KokoroTTS()
        self.asr = asr if asr is not None else WhisperASR()
        self.voice = voice
        self.t2i = t2i if t2i is not None else SDXLTurboT2I()
        self.clip = clip if clip is not None else TransformersCLIPScorer()
        self.fps = fps

    # ---- Phase 2: narration ---------------------------------------------

    def narrate(self, beat: Beat, spec: ProductionSpec) -> NarrationOut:
        voice = self.voice or spec.episode.voice.voice
        samples, sr = self.tts.synth(beat.narration, voice=voice)
        path = self.out_dir / f"{beat.id}.wav"
        write_wav(path, samples, sr)
        duration_s = (len(samples) / sr) if sr else 0.0
        transcript = self.asr.transcribe(str(path))
        return NarrationOut(audio_path=str(path), transcript=transcript, duration_s=duration_s)

    # ---- Phase 3: visuals -----------------------------------------------

    def still(self, beat: Beat, spec: ProductionSpec) -> StillOut:
        """Generate a still with SDXL-turbo, score it with CLIP."""
        path = self.out_dir / f"{beat.id}.png"
        self.t2i.generate(beat.footage_prompt, path)
        score = self.clip.score(str(path), beat.footage_prompt)
        return StillOut(image_path=str(path), clip_score=score)

    def motion(self, beat: Beat, still: StillOut, spec: ProductionSpec) -> ClipOut:
        """Ken Burns pan over the still → H.264 clip at episode resolution."""
        W, H = spec.episode.resolution
        beat_idx = next((i for i, b in enumerate(spec.beats) if b.id == beat.id), 0)
        vf = _ken_burns_filter(beat_idx, W, H, beat.duration_s)
        out_path = self.out_dir / f"{beat.id}_clip.mp4"
        cmd = [
            "ffmpeg", "-loop", "1", "-i", str(still.image_path),
            "-vf", vf,
            "-t", str(beat.duration_s),
            "-r", str(self.fps),
            "-c:v", "libx264", "-preset", "ultrafast", "-tune", "stillimage",
            "-y", str(out_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        dur = _ffprobe_duration(out_path)
        return ClipOut(clip_path=str(out_path), duration_s=dur)

    # ---- Phase 4/5: not yet implemented ---------------------------------

    def music(self, beat: Beat, spec: ProductionSpec) -> MusicOut | None:
        raise NotImplementedError("music(): Phase 4 — MusicGen / Stable Audio")

    def assemble(self, clips, audio_path: str, spec: ProductionSpec) -> ProbeOut:
        raise NotImplementedError("assemble(): Phase 5 — ffmpeg grade + mix + ffprobe")
