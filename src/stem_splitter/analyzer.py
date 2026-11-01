"""Audio analysis module for technical metadata, loudness metrics, BPM, and musical key."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np
from mutagen.id3 import ID3
from mutagen.mp3 import MP3

MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

CAMELOT_MAP = {
    "C Major": "8B", "C# Major": "3B", "D Major": "10B", "D# Major": "5B",
    "E Major": "12B", "F Major": "7B", "F# Major": "2B", "G Major": "9B",
    "G# Major": "4B", "A Major": "11B", "A# Major": "6B", "B Major": "1B",
    "C Minor": "5A", "C# Minor": "12A", "D Minor": "7A", "D# Minor": "2A",
    "E Minor": "9A", "F Minor": "4A", "F# Minor": "11A", "G Minor": "6A",
    "G# Minor": "1A", "A Minor": "8A", "A# Minor": "3A", "B Minor": "10A",
}

ID3_TAG_FRAMES = {
    "TIT2": "title", "TPE1": "artist", "TALB": "album",
    "TCON": "genre", "TBPM": "id3_bpm", "TKEY": "id3_key",
}


@dataclass
class TrackAnalysis:
    """Structured analysis result for an input audio file."""

    file_name: str
    file_path: str
    file_size_mb: float
    duration_seconds: float
    duration_formatted: str
    sample_rate_hz: int
    channels: int
    channel_layout: str
    bitrate_kbps: int
    codec: str
    peak_dbfs: float
    rms_dbfs: float
    crest_factor_db: float
    bpm: float
    musical_key: str
    camelot_key: str
    key_confidence: float
    tags: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def format_duration(seconds: float) -> str:
    """Format seconds as MM:SS or HH:MM:SS."""
    mins, secs = divmod(int(round(seconds)), 60)
    hrs, mins = divmod(mins, 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}" if hrs else f"{mins:02d}:{secs:02d}"


def _probe_technical_metadata(audio_path: Path) -> dict[str, Any]:
    """Extract technical stream metadata via Mutagen and/or ffprobe."""
    info: dict[str, Any] = {
        "duration": 0.0,
        "sample_rate": 44100,
        "channels": 2,
        "bitrate_kbps": 320,
        "codec": audio_path.suffix.lstrip(".").upper() or "MP3",
        "tags": {},
    }

    if audio_path.suffix.lower() == ".mp3":
        try:
            mp3 = MP3(audio_path)
            if mp3.info:
                info.update(
                    duration=float(getattr(mp3.info, "length", 0.0)),
                    sample_rate=int(getattr(mp3.info, "sample_rate", 44100)),
                    channels=int(getattr(mp3.info, "channels", 2)),
                    bitrate_kbps=int(round(getattr(mp3.info, "bitrate", 320000) / 1000.0)),
                    codec="MP3 (MPEG-1 Audio Layer III)",
                )
            id3 = ID3(audio_path)
            info["tags"] = {
                label: str(id3[fid].text[0])
                for fid, label in ID3_TAG_FRAMES.items()
                if fid in id3 and id3[fid].text
            }
        except Exception:
            pass

    if info["duration"] <= 0:
        try:
            cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", str(audio_path)]
            data = json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)
            stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), {})
            fmt = data.get("format", {})
            info["sample_rate"] = int(stream.get("sample_rate", info["sample_rate"]))
            info["channels"] = int(stream.get("channels", info["channels"]))
            info["codec"] = stream.get("codec_name", info["codec"]).upper()
            info["duration"] = float(fmt.get("duration", info["duration"]))
            raw_br = stream.get("bit_rate") or fmt.get("bit_rate")
            if raw_br:
                info["bitrate_kbps"] = int(round(int(raw_br) / 1000.0))
        except Exception:
            pass

    return info


def _detect_musical_key(y_harmonic: np.ndarray, sr: int) -> tuple[str, str, float]:
    """Detect musical key and Camelot notation using Chroma CENS and Krumhansl-Kessler profiles."""
    chroma_vals = np.mean(librosa.feature.chroma_cens(y=y_harmonic, sr=sr), axis=1)
    if np.allclose(chroma_vals, 0):
        return "C Major", "8B", 0.0

    norm = lambda v: (v - np.mean(v)) / (np.std(v) + 1e-8)
    c_norm, maj_norm, min_norm = norm(chroma_vals), norm(MAJOR_PROFILE), norm(MINOR_PROFILE)

    candidates = [
        (float(np.dot(np.roll(c_norm, -i), prof) / 12.0), f"{pitch} {mode}")
        for i, pitch in enumerate(PITCH_CLASSES)
        for prof, mode in ((maj_norm, "Major"), (min_norm, "Minor"))
    ]
    best_score, best_key = max(candidates, key=lambda item: item[0])
    confidence = round(max(0.0, min(1.0, (best_score + 1.0) / 2.0)), 3)
    return best_key, CAMELOT_MAP.get(best_key, "Unknown"), confidence


def analyze_audio(audio_path: Path, max_analysis_duration: float = 180.0) -> TrackAnalysis:
    """Perform full technical, loudness, BPM, and musical key analysis on an audio file."""
    audio_path = audio_path.resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    meta = _probe_technical_metadata(audio_path)
    total_dur = meta["duration"]
    offset = 15.0 if total_dur > max_analysis_duration + 30.0 else 0.0
    load_dur = max_analysis_duration if offset > 0 else None

    y, sr = librosa.load(str(audio_path), sr=22050, mono=True, offset=offset, duration=load_dur)
    if total_dur <= 0:
        total_dur = float(librosa.get_duration(y=y, sr=sr))

    peak_amp = float(np.max(np.abs(y))) if y.size else 0.0
    rms_amp = float(np.sqrt(np.mean(np.square(y)))) if y.size else 0.0
    peak_dbfs = round(20.0 * np.log10(max(peak_amp, 1e-10)), 2)
    rms_dbfs = round(20.0 * np.log10(max(rms_amp, 1e-10)), 2)

    y_harm, y_perc = librosa.effects.hpss(y)
    tempo, _ = librosa.beat.beat_track(y=y_perc, sr=sr)
    bpm = round(float(tempo[0] if isinstance(tempo, np.ndarray) and tempo.size else tempo), 1)
    musical_key, camelot_key, key_conf = _detect_musical_key(y_harm, sr)

    ch = meta["channels"]
    return TrackAnalysis(
        file_name=audio_path.name,
        file_path=str(audio_path),
        file_size_mb=round(audio_path.stat().st_size / (1024 * 1024), 2),
        duration_seconds=round(total_dur, 2),
        duration_formatted=format_duration(total_dur),
        sample_rate_hz=meta["sample_rate"],
        channels=ch,
        channel_layout={1: "Mono", 2: "Stereo"}.get(ch, f"{ch}ch"),
        bitrate_kbps=meta["bitrate_kbps"],
        codec=meta["codec"],
        peak_dbfs=peak_dbfs,
        rms_dbfs=rms_dbfs,
        crest_factor_db=round(peak_dbfs - rms_dbfs, 2),
        bpm=bpm,
        musical_key=musical_key,
        camelot_key=camelot_key,
        key_confidence=key_conf,
        tags=meta["tags"],
    )
