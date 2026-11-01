# Stem Splitter (`stem-splitter`)

An open-source CLI tool that analyzes MP3/audio tracks (technical metrics, loudness, BPM, and musical key + Camelot notation) and separates them into highest-quality **320 kbps CBR MP3 stems** using **Meta's Demucs v4 (`htdemucs`)** and **LAME (`preset=2`)**.

## Features

- **Comprehensive Pre-Separation Audio Analysis (`librosa` + `mutagen`):**
  - **Technical & Signal Metrics:** Duration, sample rate, bitrate, channel layout, Peak dBFS, RMS loudness (dBFS), and Crest Factor (dB).
  - **Musical & DJ Analysis:** Harmonic-Percussive Source Separation (HPSS) for accurate **Tempo (BPM)** tracking and **Musical Key + Camelot Wheel** detection (e.g., `8A` / `A Minor`), ready for Rekordbox, Serato, or Traktor workflows.
- **State-of-the-Art Open-Source Separation (`Demucs v4`):**
  - **4-Stem Mode (Default `htdemucs`):** `vocals.mp3`, `drums.mp3`, `bass.mp3`, `other.mp3`
  - **6-Stem Mode (`--six-stems` / `htdemucs_6s`):** Adds `guitar.mp3` and `piano.mp3`
  - **Fine-Tuned Mode (`--model htdemucs_ft`):** Bag of 4 fine-tuned models for maximum Signal-to-Distortion Ratio (SDR)
  - **2-Stem / Karaoke / Acapella Mode (`--two-stems vocals`):** Outputs `vocals.mp3` and `no_vocals.mp3` (or any chosen stem vs. instrumental sum)
- **Highest-Quality MP3 Output & Automatic ID3 Tagging:**
  - Encoded directly at **320 kbps CBR** using LAME quality preset `2` (highest quality) with automatic anti-clipping (`rescale`).
  - Automatically writes ID3v2 tags (`TIT2` Title with stem name, `TPE1` Artist, `TALB` Album, `TBPM` detected BPM, and `TKEY` Camelot Key) into every generated stem MP3.

---

## Quick Start

This project uses [`uv`](https://github.com/astral-sh/uv) for fast, isolated Python 3.11 dependency management, plus **FFmpeg** on your system PATH.

### 1. Run via `uv run` (Inside Repository)

```powershell
# Analyze and separate a track into 4 stems (vocals, drums, bass, other)
uv run stem-splitter path\to\track.mp3

# Separate into 6 stems (vocals, drums, bass, guitar, piano, other)
uv run stem-splitter path\to\track.mp3 --six-stems

# Split only Vocals + Instrumental (no_vocals.mp3)
uv run stem-splitter path\to\track.mp3 --two-stems vocals

# Specify a custom output folder and export analysis.json alongside the stems
uv run stem-splitter path\to\track.mp3 -o .\my_stems --save-json

# Run only the technical + BPM/Key analysis (no stem separation)
uv run stem-splitter path\to\track.mp3 --analyze-only

# Batch-process an entire folder of MP3 files
uv run stem-splitter path\to\folder_of_mp3s -o .\batch_stems
```

### 2. Install Globally as a Command-Line Tool (Optional)

If you want `stem-splitter` available from any folder in PowerShell without prefixing `uv run`:

```powershell
uv tool install --editable .
```

Then run `stem-splitter <track.mp3>` from anywhere.

---

## CLI Options Reference

```text
Usage: stem-splitter [OPTIONS] INPUT_PATH

Options:
  -o, --output-dir PATH           Base directory where separated MP3 stems
                                  will be saved.  [default: output]
  -m, --model [htdemucs|htdemucs_ft|htdemucs_6s|mdx_extra]
                                  Demucs v4 model: htdemucs (4 stems),
                                  htdemucs_ft (4 stems, fine-tuned),
                                  htdemucs_6s (6 stems: +guitar, +piano).
                                  [default: htdemucs]
  --six-stems                     Shortcut for --model htdemucs_6s (separates
                                  vocals, drums, bass, guitar, piano, other).
  --two-stems [vocals|drums|bass|other|guitar|piano]
                                  Only split into <STEM>.mp3 and no_<STEM>.mp3
                                  (e.g., --two-stems vocals for acapella +
                                  instrumental).
  --shifts INTEGER RANGE          Number of random time-shifts for equivariant
                                  stabilization (higher = better quality,
                                  slower).  [default: 1; 0<=x<=10]
  --overlap FLOAT RANGE           Overlap ratio between split audio segments.
                                  [default: 0.25; 0.05<=x<=0.9]
  -b, --bitrate [192|256|320]     Output MP3 constant bitrate (kbps).
                                  [default: 320]
  --analyze-only                  Only run technical and musical (BPM/Key)
                                  analysis without separating stems.
  --save-json                     Save an analysis.json file alongside the
                                  separated MP3 stems.
  -d, --device [auto|cpu|cuda]    Compute device for Demucs inference.
                                  [default: auto]
  -j, --jobs INTEGER              Number of parallel CPU threads (0 = auto).
                                  [default: 0]
  -h, --help                      Show this message and exit.
```
