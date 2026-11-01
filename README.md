# Stem Splitter (`stems` / `stem-splitter`)

A **100% local, offline-capable** open-source CLI tool that analyzes MP3/audio tracks (technical metrics, loudness, BPM, and musical key + Camelot notation) and separates them into highest-quality **320 kbps CBR MP3 stems** using **Meta's Demucs v4 (`htdemucs`)** and **LAME (`preset=2`)**.

> **Privacy & Offline Guarantee:** All audio analysis (`librosa`) and neural network stem separation (`PyTorch` + `Demucs v4`) execute **100% locally on your hardware**. Pre-trained `.safetensors` neural network weights are stored directly inside the local `models/` folder and loaded from disk without contacting any external server or requiring any API tokens.

---

## Quick Usage (`stems`)

Both `stems` (short command) and `stem-splitter` are installed and can be used interchangeably from any folder:

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

## Local Model Weights (`./models`)

Neural network weight files live directly in `C:\Users\Dario\repos\stem-splitter\models\`:
- `htdemucs.yaml` + `955717e8.safetensors` (~80 MB — Default 4-stem model)
- `htdemucs_6s.yaml` + `5c90dfd2.safetensors` (~80 MB — 6-stem model adding Guitar & Piano)

When `stems` runs, `LocalSeparator` loads `.safetensors` directly from `./models/` with zero HTTP requests. If you ever want to pre-cache all optional models (`htdemucs_ft`) into `./models/` for offline use, run:

```powershell
stems --download-models
```

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
| | `--download-models` | Pre-download local weights for models into `./models` for offline use. |
| `-h` | `--help` | Show CLI help message. |
