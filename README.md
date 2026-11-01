# Stem Splitter (`stems` / `stem-splitter`)

An open-source CLI tool that analyzes MP3/audio tracks (technical metrics, loudness, BPM, and musical key + Camelot notation) and separates them into highest-quality **320 kbps CBR MP3 stems** using **Meta's Demucs v4 (`htdemucs`)** and **LAME (`preset=2`)**.

## Quick Usage (`stems`)

Both `stems` (short command) and `stem-splitter` are installed and can be used interchangeably:

```powershell
# 1. Split a track into 4 stems (vocals, drums, bass, other) at 320 kbps MP3
stems song.mp3

# 2. Interactive mode (auto-detects MP3s in current folder or prompts for path/drag-and-drop)
stems

# 3. Split into 6 stems (+ guitar, piano)
stems song.mp3 -6

# 4. Split into Vocals + Instrumental only (vocals.mp3 + no_vocals.mp3)
stems song.mp3 -v

# 5. Isolate any single stem + accompaniment (e.g., drums.mp3 + no_drums.mp3)
stems song.mp3 -s drums

# 6. Use the fine-tuned high-quality model (htdemucs_ft) and export analysis.json
stems song.mp3 -q -j

# 7. Analyze BPM, Camelot Key & Loudness only (no stem separation)
stems song.mp3 -a

# 8. Batch-process a folder of MP3s into a custom output folder
stems .\my_tracks -o .\stems_out
```

---

## Features

- **Comprehensive Pre-Separation Audio Analysis (`librosa` + `mutagen`):**
  - **Technical & Signal Metrics:** Duration, sample rate, bitrate, channel layout, Peak dBFS, RMS loudness (dBFS), and Crest Factor (dB).
  - **Musical & DJ Analysis:** Harmonic-Percussive Source Separation (HPSS) for accurate **Tempo (BPM)** tracking and **Musical Key + Camelot Wheel** detection (e.g., `8A` / `A Minor`), ready for Rekordbox, Serato, or Traktor workflows.
- **State-of-the-Art Open-Source Separation (`Demucs v4`):**
  - **4-Stem Mode (Default `htdemucs`):** `vocals.mp3`, `drums.mp3`, `bass.mp3`, `other.mp3`
  - **6-Stem Mode (`-6` / `--six-stems`):** Adds `guitar.mp3` and `piano.mp3`
  - **Fine-Tuned HQ Mode (`-q` / `--hq`):** Bag of 4 fine-tuned models (`htdemucs_ft`) for maximum Signal-to-Distortion Ratio (SDR)
  - **2-Stem / Karaoke / Acapella Mode (`-v` or `-s <stem>`):** Outputs `<stem>.mp3` and `no_<stem>.mp3`
- **Highest-Quality MP3 Output & Automatic ID3 Tagging:**
  - Encoded directly at **320 kbps CBR** using LAME quality preset `2` (highest quality) with automatic anti-clipping (`rescale`).
  - Automatically writes ID3v2 tags (`TIT2` Title with stem name, `TPE1` Artist, `TALB` Album, `TBPM` detected BPM, and `TKEY` Camelot Key) into every generated stem MP3.

---

## CLI Options Reference

| Short | Long | Description |
| :--- | :--- | :--- |
| *(none)* | `[INPUT_PATH]` | File or folder path. Optional—if omitted, runs interactive picker. |
| `-6` | `--six-stems` | Split into 6 stems (`vocals`, `drums`, `bass`, `guitar`, `piano`, `other`). |
| `-v` | `--vocals` | Split only into `vocals.mp3` + `no_vocals.mp3` (acapella & instrumental). |
| `-s` | `--stem <name>` | Isolate any single `<stem>.mp3` + `no_<stem>.mp3`. |
| `-q` | `--hq` | Use fine-tuned `htdemucs_ft` model for maximum quality. |
| `-a` | `--analyze` | Only run BPM, Key, and Loudness analysis (skip stem separation). |
| `-j` | `--json` | Save `analysis.json` alongside the separated stems. |
| `-o` | `--output-dir <dir>` | Base output folder (default: `./output`). |
| `-m` | `--model <name>` | Model choice (`htdemucs`, `htdemucs_ft`, `htdemucs_6s`, `mdx_extra`). |
| `-b` | `--bitrate <kbps>` | Output MP3 CBR bitrate: `192`, `256`, or `320` (default: `320`). |
| `-d` | `--device <dev>` | Compute device: `auto`, `cpu`, or `cuda` (default: `auto`). |
| `-t` | `--threads <num>` | Number of CPU worker threads (`0` = auto). |
| `-h` | `--help` | Show CLI help message. |
