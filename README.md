# Stem Splitter (`stems` / `stem-splitter`)

A **100% local, offline-capable** open-source CLI tool that analyzes MP3/audio tracks (technical metrics, loudness, BPM, and musical key + Camelot notation) and separates them into highest-quality **320 kbps CBR MP3 stems** using **Meta's Demucs v4 (`htdemucs`)** and **LAME (`preset=2`)**.

> **Default Behavior:** Running `stems song.mp3` automatically separates the track into **2 MP3 files**: **`vocals.mp3`** (acapella) and **`instrumental.mp3`** (the rest of the track). You can pass `-4` for 4 stems, `-6` for 6 stems, or `-s <stem>` to isolate any specific instrument.

---

## Installation on Any Device

Clone the repository and run the included installer script for your OS. It automatically checks/installs `uv` and `ffmpeg`, sets up the Python 3.11 environment, registers `stems` on your `PATH`, and caches the local Demucs v4 neural network weights (`htdemucs` 4-stem and `htdemucs_6s` 6-stem) into `./models/` for offline use.

### Windows (PowerShell or Command Prompt)

```powershell
git clone https://github.com/dario-cruz/stem-splitter.git
cd stem-splitter

# Global install (Default — makes 'stems' available from any folder)
.\install.ps1

# Or Local-only install (installs inside .venv only)
.\install.ps1 -Mode local
```
*(You can also simply double-click **`install.bat`** in File Explorer!)*

### Linux / macOS / WSL

```bash
git clone https://github.com/dario-cruz/stem-splitter.git
cd stem-splitter
chmod +x install.sh

# Global install (Default)
./install.sh

# Or Local-only install (.venv only)
./install.sh --local
```

---

## Quick Usage (`stems`)

Both `stems` (short command) and `stem-splitter` are installed and can be used interchangeably from any folder:

```powershell
# 1. Default: Split into 2 tracks (vocals.mp3 + instrumental.mp3) at 320 kbps MP3
stems song.mp3

# 2. Interactive mode (auto-detects MP3s in current folder or prompts for path/drag-and-drop)
stems

# 3. Split into 4 stems (vocals.mp3, drums.mp3, bass.mp3, other.mp3)
stems song.mp3 -4

# 4. Split into 6 stems (vocals, drums, bass, guitar, piano, other)
stems song.mp3 -6

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

Neural network weight files live directly in `./models/` (or `~/.stem-splitter/models` if installed outside a repo checkout):
- `htdemucs.yaml` + `955717e8.safetensors` (~80 MB — Default model)
- `htdemucs_6s.yaml` + `5c90dfd2.safetensors` (~52 MB — 6-stem model adding Guitar & Piano)

When `stems` runs, `LocalSeparator` loads `.safetensors` directly from `./models/` with zero HTTP requests.

---

## CLI Options Reference

| Short | Long | Description |
| :--- | :--- | :--- |
| *(none)* | `[INPUT_PATH]` | File or folder path. Optional—if omitted, runs interactive picker. **Defaults to 2 tracks (`vocals.mp3` + `instrumental.mp3`).** |
| `-4` | `--four-stems` | Split into 4 stems (`vocals`, `drums`, `bass`, `other`). |
| `-6` | `--six-stems` | Split into 6 stems (`vocals`, `drums`, `bass`, `guitar`, `piano`, `other`). |
| `-v` | `--vocals` | Explicit flag for the default 2-track split (`vocals.mp3` + `instrumental.mp3`). |
| `-s` | `--stem <name>` | Isolate any single `<stem>.mp3` + accompaniment (`no_<stem>.mp3`). |
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
