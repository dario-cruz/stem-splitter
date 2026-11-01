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

# Krumhansl-Kessler key profiles (normalized correlation templates)
MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    dtype=np.float64,
)
MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    dtype=np.float64,
)

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Standard Camelot wheel mapping for DJ workflows (Rekordbox / Serato compatible)
CAMELOT_MAP: dict[str, str] = {
    "C Major": "8B",
    "C# Major": "3B",
    "D Major": "10B",
    "D# Major": "5B",
    "E Major": "12B",
    "F Major": "7B",
    "F# Major": "2B",
    "G Major": "9B",
    "G# Major": "4B",
    "A Major": "11B",
    "A# Major": "6B",
    "B Major": "1B",
    "C Minor": "5A",
    "C# Minor": "12A",
    "D Minor": "7A",
    "D# Minor": "2A",
    "E Minor": "9A",
    "F Minor": "4A",
    "F# Minor": "11A",
    "G Minor": "6A",
    "G# Minor": "1A",
    "A Minor": "8A",
    "A# Minor": "3A",
    "B Minor": "10A",
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
    """Format seconds as MM:SS.ms or HH:MM:SS."""
    total_sec = int(round(seconds))
    mins, secs = divmod(total_sec, 60)
    hrs, mins = divmod(mins, 60)
    if hrs > 0:
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


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

    try:
        if audio_path.suffix.lower() == ".mp3":
            mp3 = MP3(audio_path)
            if mp3.info:
                info["duration"] = float(getattr(mp3.info, "length", 0.0))
                info["sample_rate"] = int(getattr(mp3.info, "sample_rate", 44100))
                info["channels"] = int(getattr(mp3.info, "channels", 2))
                bitrate = getattr(mp3.info, "bitrate", 320000)
                info["bitrate_kbps"] = int(round(bitrate / 1000.0))
                info["codec"] = "MP3 (MPEG-1 Audio Layer III)"
            try:
                id3 = ID3(audio_path)
                tag_map = {
                    "TIT2": "title",
                    "TPE1": "artist",
                    "TALB": "album",
                    "TCON": "genre",
                    "TBPM": "id3_bpm",
                    "TKEY": "id3_key",
                }
                for frame_id, label in tag_map.items():
                    if frame_id in id3 and id3[frame_id].text:
                        info["tags"][label] = str(id3[frame_id].text[0])
            except Exception:
                pass
    except Exception:
        pass

    # Fallback / supplement with ffprobe if needed
    if info["duration"] <= 0:
        try:
            cmd = [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(audio_path),
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(res.stdout)
            for stream in data.get("streams", []):
                if stream.get("codec_type") == "audio":
                    info["sample_rate"] = int(stream.get("sample_rate", info["sample_rate"]))
                    info["channels"] = int(stream.get("channels", info["channels"]))
                    info["codec"] = stream.get("codec_name", info["codec"]).upper()
                    if "bit_rate" in stream:
                        info["bitrate_kbps"] = int(round(int(stream["bit_rate"]) / 1000.0))
                    break
            fmt = data.get("format", {})
            if "duration" in fmt:
                info["duration"] = float(fmt["duration"])
            if "bit_rate" in fmt and info["bitrate_kbps"] == 320:
                info["bitrate_kbps"] = int(round(int(fmt["bit_rate"]) / 1000.0))
        except Exception:
            pass

    return info


def _detect_musical_key(y_harmonic: np.ndarray, sr: int) -> tuple[str, str, float]:
    """Detect musical key and Camelot notation using Chroma CENS and Krumhansl-Kessler profiles."""
    chroma = librosa.feature.chroma_cens(y=y_harmonic, sr=sr)
    chroma_vals = np.mean(chroma, axis=1)

    if np.allclose(chroma_vals, 0):
        return "C Major", "8B", 0.0

    # Normalize chroma vector
    chroma_norm = (chroma_vals - np.mean(chroma_vals)) / (np.std(chroma_vals) + 1e-8)
    maj_norm = (MAJOR_PROFILE - np.mean(MAJOR_PROFILE)) / np.std(MAJOR_PROFILE)
    min_norm = (MINOR_PROFILE - np.mean(MINOR_PROFILE)) / np.std(MINOR_PROFILE)

    best_score = -2.0
    best_key = "C Major"

    for i, pitch in enumerate(PITCH_CLASSES):
        rotated_chroma = np.roll(chroma_norm, -i)
        maj_corr = float(np.dot(rotated_chroma, maj_norm) / 12.0)
        min_corr = float(np.dot(rotated_chroma, min_norm) / 12.0)

        if maj_corr > best_score:
            best_score = maj_corr
            best_key = f"{pitch} Major"
        if min_corr > best_score:
            best_score = min_corr
            best_key = f"{pitch} Minor"

    camelot = CAMELOT_MAP.get(best_key, "Unknown")
    confidence = max(0.0, min(1.0, (best_score + 1.0) / 2.0))
    return best_key, camelot, round(confidence, 3)


def analyze_audio(audio_path: Path, max_analysis_duration: float = 180.0) -> TrackAnalysis:
    """Perform full technical, loudness, BPM, and musical key analysis on an audio file.

    Args:
        audio_path: Path to the input MP3/audio file.
        max_analysis_duration: Maximum seconds of audio from the core of the track
            to use for BPM/key estimation (improves speed on long mixes while maintaining accuracy).
    """
    audio_path = audio_path.resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    meta = _probe_technical_metadata(audio_path)
    file_size_mb = round(audio_path.stat().st_size / (1024 * 1024), 2)

    # Load full signal at native or 22050 Hz for fast & accurate DSP metrics
    # Choose an offset/window if the track is very long, but compute peak/RMS cleanly
    total_duration = meta["duration"]
    offset = 0.0
    duration_to_load = None
    if total_duration > max_analysis_duration + 30.0:
        # Skip intro (15s) and analyze up to max_analysis_duration
        offset = 15.0
        duration_to_load = max_analysis_duration

    y, sr = librosa.load(
        str(audio_path),
        sr=22050,
        mono=True,
        offset=offset,
        duration=duration_to_load,
    )

    if total_duration <= 0:
        total_duration = float(librosa.get_duration(y=y, sr=sr))

    # Signal loudness stats (Peak dBFS, RMS dBFS, Crest Factor)
    eps = 1e-10
    peak_amp = float(np.max(np.abs(y))) if y.size > 0 else 0.0
    rms_amp = float(np.sqrt(np.mean(np.square(y)))) if y.size > 0 else 0.0

    peak_dbfs = round(20.0 * np.log10(max(peak_amp, eps)), 2)
    rms_dbfs = round(20.0 * np.log10(max(rms_amp, eps)), 2)
    crest_factor_db = round(peak_dbfs - rms_dbfs, 2)

    # Harmonic-percussive separation for cleaner BPM and Key detection
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    # BPM estimation using percussive component
    tempo, _ = librosa.beat.beat_track(y=y_percussive, sr=sr)
    if isinstance(tempo, np.ndarray):
        bpm = float(tempo[0]) if tempo.size > 0 else 0.0
    else:
        bpm = float(tempo)
    bpm = round(bpm, 1)

    # Key detection using harmonic component
    musical_key, camelot_key, key_confidence = _detect_musical_key(y_harmonic, sr)

    channels = meta["channels"]
    channel_layout = "Stereo" if channels == 2 else ("Mono" if channels == 1 else f"{channels}ch")

    return TrackAnalysis(
        file_name=audio_path.name,
        file_path=str(audio_path),
        file_size_mb=file_size_mb,
        duration_seconds=round(total_duration, 2),
        duration_formatted=format_duration(total_duration),
        sample_rate_hz=meta["sample_rate"],
        channels=channels,
        channel_layout=channel_layout,
        bitrate_kbps=meta["bitrate_kbps"],
        codec=meta["codec"],
        peak_dbfs=peak_dbfs,
        rms_dbfs=rms_dbfs,
        crest_factor_db=crest_factor_db,
        bpm=bpm,
        musical_key=musical_key,
        camelot_key=camelot_key,
        key_confidence=key_confidence,
        tags=meta["tags"],
    )
