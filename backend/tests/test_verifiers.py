from myAIscene import verifiers as V
from myAIscene.spec import Beat


def mkbeat(t0=0.0, t1=10.0, narration="the quick brown fox jumps over the lazy dog"):
    return Beat(id="b1", t0=t0, t1=t1, narration=narration, direction="d", footage_prompt="f")


def test_narration_exact_match_passes():
    b = mkbeat()
    assert V.narration_verify(b, b.narration).passed


def test_narration_minor_punctuation_still_passes():
    b = mkbeat(narration="It is swift. It is ruthless.")
    r = V.narration_verify(b, "it is swift it is ruthless")
    assert r.passed


def test_narration_wrong_words_fail_and_blocking():
    b = mkbeat()
    r = V.narration_verify(b, "completely unrelated transcription output here")
    assert not r.passed and r.blocking


def test_duration_underflow_ok_overflow_fails():
    b = mkbeat(t0=0, t1=10)
    assert V.duration_verify(b, 9.0).passed          # under window: fine
    assert V.duration_verify(b, 10.4).passed         # within tol
    assert not V.duration_verify(b, 11.0).passed     # overflow beyond tol


def test_footage_clip_threshold():
    b = mkbeat()
    assert V.footage_verify(b, 0.30).passed
    assert not V.footage_verify(b, 0.10).passed
    assert V.footage_verify(b, 0.10).blocking


def test_music_cue_is_non_blocking():
    b = mkbeat(t0=0, t1=10)
    ok = V.music_cue_verify(b, "/x.wav", 10.5)
    assert ok.passed and not ok.blocking
    missing = V.music_cue_verify(b, None, None)
    assert not missing.passed and not missing.blocking  # fails soft
    short = V.music_cue_verify(b, "/x.wav", 4.0)
    assert not short.passed and not short.blocking


def test_assembly_verify():
    ok = V.assembly_verify(exists=True, duration_s=180.0, expected_s=180.3,
                           has_audio=True, resolution=(1920, 1080),
                           expected_resolution=(1920, 1080))
    assert ok.passed
    bad = V.assembly_verify(exists=True, duration_s=170.0, expected_s=180.0,
                            has_audio=False, resolution=(1280, 720),
                            expected_resolution=(1920, 1080))
    assert not bad.passed and bad.blocking
    assert "no audio track" in bad.detail
