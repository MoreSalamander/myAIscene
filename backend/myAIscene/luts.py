"""Procedural .cube LUT generator — no paid assets, no downloads.

Named grades match the spec grades in not_it_protocol.json (§3 visual
style). Generated at LUT_SIZE³ resolution, written as standard .cube
files readable by ffmpeg's lut3d filter. Idempotent: only writes if
the file doesn't exist yet.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

LUT_SIZE = 17  # 17³ = 4913 entries — smooth enough for cinematic grading


# ---- colour transforms -------------------------------------------------
# All take/return float32 numpy arrays in [0, 1].

def _identity(r, g, b):
    return r, g, b


def _warm_documentary(r, g, b):
    r = np.clip(r * 1.05 + 0.02, 0, 1)
    g = np.clip(g * 1.01, 0, 1)
    b = np.clip(b * 0.92, 0, 1)
    return r, g, b


def _cool_neutral(r, g, b):
    r = np.clip(r * 0.97, 0, 1)
    b = np.clip(b * 1.05, 0, 1)
    return r, g, b


def _high_contrast(r, g, b):
    def scurve(x):
        s = 1.0 / (1.0 + np.exp(-10.0 * (x - 0.5)))
        lo, hi = 1.0 / (1.0 + np.exp(5.0)), 1.0 / (1.0 + np.exp(-5.0))
        return np.clip((s - lo) / (hi - lo), 0, 1)
    return scurve(r), scurve(g), scurve(b)


def _desaturated(r, g, b):
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return (np.clip(r * 0.5 + lum * 0.5, 0, 1),
            np.clip(g * 0.5 + lum * 0.5, 0, 1),
            np.clip(b * 0.5 + lum * 0.5, 0, 1))


def _cool_vignette(r, g, b):
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    shadow = np.clip(1.0 - lum * 2.0, 0, 1)
    return np.clip(r * 0.97, 0, 1), g, np.clip(b + shadow * 0.10, 0, 1)


def _clean_sterile(r, g, b):
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    r = r * 0.80 + lum * 0.20
    g = g * 0.80 + lum * 0.20
    b = b * 0.80 + lum * 0.20
    return (np.clip(r * 0.90 + 0.05, 0, 1),
            np.clip(g * 0.90 + 0.05, 0, 1),
            np.clip(b * 0.90 + 0.05, 0, 1))


def _warm_orange(r, g, b):
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    hi, sh = lum, 1.0 - lum
    return (np.clip(r + hi * 0.08 - sh * 0.03, 0, 1),
            np.clip(g + hi * 0.03, 0, 1),
            np.clip(b - hi * 0.05 + sh * 0.08, 0, 1))


def _low_contrast(r, g, b):
    return (np.clip(r * 0.75 + 0.10, 0, 1),
            np.clip(g * 0.75 + 0.10, 0, 1),
            np.clip(b * 0.75 + 0.10, 0, 1))


def _bright_vivid(r, g, b):
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return (np.clip((r - lum) * 1.3 + lum + 0.05, 0, 1),
            np.clip((g - lum) * 1.3 + lum + 0.05, 0, 1),
            np.clip((b - lum) * 1.3 + lum + 0.05, 0, 1))


def _sepia_fade(r, g, b):
    sr = np.clip(r * 0.393 + g * 0.769 + b * 0.189, 0, 1)
    sg = np.clip(r * 0.349 + g * 0.686 + b * 0.168, 0, 1)
    sb = np.clip(r * 0.272 + g * 0.534 + b * 0.131, 0, 1)
    return (np.clip(sr * 0.85 + 0.05, 0, 1),
            np.clip(sg * 0.85 + 0.05, 0, 1),
            np.clip(sb * 0.85 + 0.05, 0, 1))


GRADES: dict[str, object] = {
    "identity":         _identity,
    "warm_documentary": _warm_documentary,
    "cool_neutral":     _cool_neutral,
    "high_contrast":    _high_contrast,
    "desaturated":      _desaturated,
    "cool_vignette":    _cool_vignette,
    "clean_sterile":    _clean_sterile,
    "warm_orange":      _warm_orange,
    "low_contrast":     _low_contrast,
    "bright_vivid":     _bright_vivid,
    "sepia_fade":       _sepia_fade,
}

DEFAULT_GRADE = "warm_documentary"


def generate_cube(grade_name: str) -> str:
    """Return a .cube LUT string for the named grade (or identity if unknown)."""
    fn = GRADES.get(grade_name, _identity)
    n = LUT_SIZE
    grid = np.linspace(0, 1, n, dtype=np.float32)
    R, G, B = np.meshgrid(grid, grid, grid, indexing="ij")
    r_out, g_out, b_out = fn(R.astype(np.float32), G.astype(np.float32), B.astype(np.float32))

    # .cube spec: R varies fastest (inner), B slowest (outer) — F-order of [r,g,b] array
    r_flat = np.clip(r_out, 0, 1).ravel(order="F")
    g_flat = np.clip(g_out, 0, 1).ravel(order="F")
    b_flat = np.clip(b_out, 0, 1).ravel(order="F")

    lines = [f"LUT_3D_SIZE {n}"]
    lines += [f"{r:.6f} {g:.6f} {b:.6f}" for r, g, b in zip(r_flat, g_flat, b_flat)]
    return "\n".join(lines) + "\n"


def ensure_luts(lut_dir: Path) -> dict[str, Path]:
    """Write all named LUT .cube files to lut_dir (idempotent). Returns name→path."""
    lut_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name in GRADES:
        p = lut_dir / f"{name}.cube"
        if not p.exists():
            p.write_text(generate_cube(name))
        paths[name] = p
    return paths


def lut_for(grade_name: str, lut_paths: dict[str, Path]) -> Path:
    """Return the LUT path for a grade name, falling back to the default."""
    return lut_paths.get(grade_name) or lut_paths[DEFAULT_GRADE]
