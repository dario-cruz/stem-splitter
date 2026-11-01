"""Demucs v4 local-first stem separation and highest-quality MP3 encoding with ID3 metadata tagging."""

from __future__ import annotations

import os
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import torch
import yaml
from demucs.api import Separator, save_audio
from demucs.apply import BagOfModels
from demucs.hf import DEFAULT_NAMESPACE, hf_repo_name, load_safetensors_model
from demucs.pretrained import REMOTE_ROOT
from mutagen.id3 import ID3, TALB, TBPM, TCON, TIT2, TKEY, TPE1

from .analyzer import TrackAnalysis

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")

# Local models folder: use repo ./models if running from source checkout, else ~/.stem-splitter/models
_REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = (
    _REPO_ROOT / "models"
    if (_REPO_ROOT / "pyproject.toml").exists()
    else Path.home() / ".stem-splitter" / "models"
)

SUPPORTED_MODELS: dict[str, str] = {
    "htdemucs": "Hybrid Transformer Demucs v4 (4 stems: vocals, drums, bass, other)",
    "htdemucs_6s": "Hybrid Transformer Demucs v4 6-stem (vocals, drums, bass, guitar, piano, other)",
    "htdemucs_ft": "Fine-tuned Hybrid Transformer Demucs v4 (4 stems, 4x slower, highest SDR)",
    "mdx_extra": "MDX-Net Extra (4 stems, strong vocal/instrumental isolation)",
}


@dataclass
class StemOutput:
    """Metadata about an exported stem MP3 file."""

    stem_name: str
    file_path: Path
    file_size_mb: float


def ensure_local_model(model_name: str, models_dir: Path = MODELS_DIR) -> list[Path]:
    """Ensure all safetensors weights for `model_name` exist in local `models_dir`.

    Only downloads missing `.safetensors` files once into `models_dir`; if files are
    already in `models_dir`, zero network calls are made.
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    yaml_src = REMOTE_ROOT / f"{model_name}.yaml"
    if not yaml_src.exists():
        raise ValueError(f"Unknown Demucs model configuration: {model_name}")

    yaml_dst = models_dir / f"{model_name}.yaml"
    if not yaml_dst.exists():
        shutil.copy2(yaml_src, yaml_dst)

    bag_cfg = yaml.safe_load(yaml_dst.read_text(encoding="utf-8"))
    signatures: list[str] = bag_cfg["models"]
    weight_paths: list[Path] = []

    for sig in signatures:
        local_file = models_dir / f"{sig}.safetensors"
        if not local_file.exists():
            # One-time fetch into local models/ directory
            from huggingface_hub import hf_hub_download

            repo_id = f"{DEFAULT_NAMESPACE}/{hf_repo_name(model_name)}"
            cached_path = Path(hf_hub_download(repo_id, f"{sig}.safetensors"))
            shutil.copy2(cached_path, local_file)
        weight_paths.append(local_file)

    return weight_paths


def load_local_bag(model_name: str, models_dir: Path = MODELS_DIR) -> BagOfModels:
    """Load a Demucs `BagOfModels` directly from local `.safetensors` files on disk (zero network)."""
    ensure_local_model(model_name, models_dir)
    bag_cfg = yaml.safe_load((models_dir / f"{model_name}.yaml").read_text(encoding="utf-8"))
    submodels = [load_safetensors_model(models_dir / f"{sig}.safetensors") for sig in bag_cfg["models"]]
    bag = BagOfModels(submodels, bag_cfg.get("weights"), bag_cfg.get("segment"))
    bag.eval()
    return bag


class LocalSeparator(Separator):
    """Subclass of `demucs.api.Separator` that loads models strictly from local disk (`models/`)."""

    def _load_model(self) -> None:
        self._model = load_local_bag(self._name)
        self._audio_channels = self._model.audio_channels
        self._samplerate = self._model.samplerate


def _tag_stem_mp3(mp3_path: Path, stem_name: str, analysis: TrackAnalysis | None) -> None:
    """Embed ID3v2 tags (including detected BPM, Musical Key, and original artist/title) into stem MP3."""
    if not analysis:
        return
    try:
        try:
            tags = ID3(mp3_path)
        except Exception:
            tags = ID3()

        base_title = analysis.tags.get("title") or Path(analysis.file_name).stem
        tags.add(TIT2(encoding=3, text=f"{base_title} ({stem_name.capitalize()} Stem)"))

        for key, frame_cls in (("artist", TPE1), ("album", TALB), ("genre", TCON)):
            if key in analysis.tags:
                tags.add(frame_cls(encoding=3, text=analysis.tags[key]))

        if analysis.bpm > 0:
            tags.add(TBPM(encoding=3, text=str(int(round(analysis.bpm)))))

        key_val = analysis.camelot_key if analysis.camelot_key != "Unknown" else analysis.musical_key
        if key_val:
            tags.add(TKEY(encoding=3, text=key_val))

        tags.save(mp3_path)
    except Exception:
        pass


def separate_to_stems(
    audio_path: Path,
    output_dir: Path,
    model_name: str = "htdemucs",
    shifts: int = 1,
    overlap: float = 0.25,
    two_stems: str | None = "vocals",
    mp3_bitrate: int = 320,
    mp3_preset: Literal[2, 3, 4, 5, 6, 7] = 2,
    clip_mode: Literal["rescale", "clamp", "tanh", "none"] = "rescale",
    device: str | None = None,
    jobs: int = 0,
    analysis: TrackAnalysis | None = None,
    progress_callback: Callable[[dict], None] | None = None,
) -> list[StemOutput]:
    """Separate an audio track into highest-quality MP3 stems using local Demucs v4 weights."""
    audio_path = audio_path.resolve()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if jobs <= 0:
        jobs = max(1, (os.cpu_count() or 4) // 2) if device == "cpu" else 0

    separator = LocalSeparator(
        model=model_name,
        device=device,
        shifts=shifts,
        overlap=overlap,
        split=True,
        jobs=jobs,
        progress=False,
        callback=progress_callback,
    )

    if two_stems and two_stems not in separator.model.sources:
        valid = ", ".join(separator.model.sources)
        raise ValueError(f"Stem '{two_stems}' is not in '{model_name}'. Valid stems: {valid}")

    _, stems_dict = separator.separate_audio_file(audio_path)
    track_dir = output_dir.resolve() / audio_path.stem
    track_dir.mkdir(parents=True, exist_ok=True)

    def _export(name: str, tensor: torch.Tensor) -> StemOutput:
        out_file = track_dir / f"{name}.mp3"
        save_audio(
            tensor,
            str(out_file),
            samplerate=separator.samplerate,
            bitrate=mp3_bitrate,
            preset=mp3_preset,
            clip=clip_mode,
        )
        _tag_stem_mp3(out_file, name, analysis)
        return StemOutput(name, out_file, round(out_file.stat().st_size / (1024 * 1024), 2))

    if not two_stems:
        return [_export(name, tensor) for name, tensor in stems_dict.items()]

    target_tensor = stems_dict.pop(two_stems)
    residual_sum = torch.stack(list(stems_dict.values())).sum(dim=0)
    second_name = "instrumental" if two_stems == "vocals" else f"no_{two_stems}"
    return [_export(two_stems, target_tensor), _export(second_name, residual_sum)]
