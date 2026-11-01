"""Demucs v4 stem separation and highest-quality MP3 encoding with ID3 metadata tagging."""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")

import torch
from demucs.api import Separator, save_audio
from mutagen.id3 import ID3, TBPM, TCON, TIT2, TKEY, TALB, TPE1

from .analyzer import TrackAnalysis

SUPPORTED_MODELS: dict[str, str] = {
    "htdemucs": "Hybrid Transformer Demucs v4 (4 stems: vocals, drums, bass, other)",
    "htdemucs_ft": "Fine-tuned Hybrid Transformer Demucs v4 (4 stems, 4x slower, highest SDR)",
    "htdemucs_6s": "Hybrid Transformer Demucs v4 6-stem (vocals, drums, bass, guitar, piano, other)",
    "mdx_extra": "MDX-Net Extra (4 stems, strong vocal/instrumental isolation)",
}


@dataclass
class StemOutput:
    """Metadata about an exported stem MP3 file."""

    stem_name: str
    file_path: Path
    file_size_mb: float


def _tag_stem_mp3(
    mp3_path: Path,
    stem_name: str,
    analysis: TrackAnalysis | None,
) -> None:
    """Embed ID3v2 tags (including detected BPM, Musical Key, and original artist/title) into stem MP3."""
    if analysis is None:
        return

    try:
        try:
            tags = ID3(mp3_path)
        except Exception:
            tags = ID3()

        base_title = analysis.tags.get("title") or Path(analysis.file_name).stem
        tags.add(TIT2(encoding=3, text=f"{base_title} ({stem_name.capitalize()} Stem)"))

        if "artist" in analysis.tags:
            tags.add(TPE1(encoding=3, text=analysis.tags["artist"]))
        if "album" in analysis.tags:
            tags.add(TALB(encoding=3, text=analysis.tags["album"]))
        if "genre" in analysis.tags:
            tags.add(TCON(encoding=3, text=analysis.tags["genre"]))

        if analysis.bpm > 0:
            tags.add(TBPM(encoding=3, text=str(int(round(analysis.bpm)))))

        # Use Camelot or standard key in TKEY frame (recognized by Rekordbox/Serato/Traktor)
        if analysis.camelot_key and analysis.camelot_key != "Unknown":
            tags.add(TKEY(encoding=3, text=analysis.camelot_key))
        elif analysis.musical_key:
            tags.add(TKEY(encoding=3, text=analysis.musical_key))

        tags.save(mp3_path)
    except Exception:
        # Non-fatal if tagging fails on edge-case file locks
        pass


def separate_to_stems(
    audio_path: Path,
    output_dir: Path,
    model_name: str = "htdemucs",
    shifts: int = 1,
    overlap: float = 0.25,
    two_stems: str | None = None,
    mp3_bitrate: int = 320,
    mp3_preset: Literal[2, 3, 4, 5, 6, 7] = 2,
    clip_mode: Literal["rescale", "clamp", "tanh", "none"] = "rescale",
    device: str | None = None,
    jobs: int = 0,
    analysis: TrackAnalysis | None = None,
    progress_callback: Callable[[dict], None] | None = None,
) -> list[StemOutput]:
    """Separate an audio track into highest-quality MP3 stems using Demucs v4.

    Args:
        audio_path: Input audio file path.
        output_dir: Directory where `<track_name>/<stem>.mp3` files will be written.
        model_name: Demucs model (`htdemucs`, `htdemucs_ft`, `htdemucs_6s`, etc.).
        shifts: Number of random time-shifts for equivariant stabilization (1 = default, >1 = higher quality).
        overlap: Overlap ratio between segments (default 0.25).
        two_stems: Optional single stem name (e.g., `vocals`) to split into `<stem>.mp3` and `no_<stem>.mp3`.
        mp3_bitrate: CBR MP3 bitrate in kbps (default 320 for maximum MP3 fidelity).
        mp3_preset: LAME encoder quality preset (2 = highest quality, 7 = fastest).
        clip_mode: Anti-clipping strategy (`rescale` or `clamp`).
        device: Torch device (`cuda`, `cpu`, or auto-detected if None).
        jobs: Number of parallel CPU worker threads.
        analysis: Optional `TrackAnalysis` to embed BPM, Key, and Title into output MP3 ID3 tags.
        progress_callback: Optional callback invoked by Demucs during segment separation.
    """
    audio_path = audio_path.resolve()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if jobs <= 0:
        cpu_count = os.cpu_count() or 4
        jobs = max(1, cpu_count // 2) if device == "cpu" else 0

    separator = Separator(
        model=model_name,
        device=device,
        shifts=shifts,
        overlap=overlap,
        split=True,
        jobs=jobs,
        progress=False,
        callback=progress_callback,
    )

    if two_stems is not None and two_stems not in separator.model.sources:
        valid = ", ".join(separator.model.sources)
        raise ValueError(
            f"Stem '{two_stems}' is not supported by model '{model_name}'. Valid stems: {valid}"
        )

    _, stems_dict = separator.separate_audio_file(audio_path)

    track_stem_dir = output_dir.resolve() / audio_path.stem
    track_stem_dir.mkdir(parents=True, exist_ok=True)

    save_kwargs = {
        "samplerate": separator.samplerate,
        "bitrate": mp3_bitrate,
        "preset": mp3_preset,
        "clip": clip_mode,
    }

    outputs: list[StemOutput] = []

    if two_stems is None:
        for stem_name, tensor in stems_dict.items():
            out_file = track_stem_dir / f"{stem_name}.mp3"
            save_audio(tensor, str(out_file), **save_kwargs)
            _tag_stem_mp3(out_file, stem_name, analysis)
            size_mb = round(out_file.stat().st_size / (1024 * 1024), 2)
            outputs.append(StemOutput(stem_name=stem_name, file_path=out_file, file_size_mb=size_mb))
    else:
        # Target stem
        target_tensor = stems_dict.pop(two_stems)
        target_file = track_stem_dir / f"{two_stems}.mp3"
        save_audio(target_tensor, str(target_file), **save_kwargs)
        _tag_stem_mp3(target_file, two_stems, analysis)
        outputs.append(
            StemOutput(
                stem_name=two_stems,
                file_path=target_file,
                file_size_mb=round(target_file.stat().st_size / (1024 * 1024), 2),
            )
        )

        # Instrumental / accompaniment sum
        accompaniment = torch.zeros_like(target_tensor)
        for residual in stems_dict.values():
            accompaniment += residual
        no_stem_name = f"no_{two_stems}"
        no_stem_file = track_stem_dir / f"{no_stem_name}.mp3"
        save_audio(accompaniment, str(no_stem_file), **save_kwargs)
        _tag_stem_mp3(no_stem_file, no_stem_name, analysis)
        outputs.append(
            StemOutput(
                stem_name=no_stem_name,
                file_path=no_stem_file,
                file_size_mb=round(no_stem_file.stat().st_size / (1024 * 1024), 2),
            )
        )

    return outputs
