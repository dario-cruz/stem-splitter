"""Rich CLI interface for Stem Splitter: 100% Local Audio Analysis & Demucs v4 Stem Separation."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn
from rich.prompt import Prompt
from rich.table import Table

from .analyzer import TrackAnalysis, analyze_audio
from .separator import MODELS_DIR, SUPPORTED_MODELS, StemOutput, ensure_local_model, separate_to_stems

console = Console()
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac"}
STEM_CHOICES = ["vocals", "drums", "bass", "other", "guitar", "piano"]


def _resolve_input_path(input_path: Path | None) -> Path:
    """Resolve input path or prompt interactively if omitted."""
    if input_path is not None:
        return input_path

    local_files = sorted(p for p in Path.cwd().iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS)
    if len(local_files) == 1:
        console.print(f"[cyan]Auto-detected audio file in current folder:[/cyan] [bold]{local_files[0].name}[/bold]")
        return local_files[0]
    if len(local_files) > 1:
        console.print("[bold cyan]Audio files in current directory:[/bold cyan]")
        for i, f in enumerate(local_files, 1):
            console.print(f"  [bold]{i}.[/bold] {f.name}")

    raw = Prompt.ask("\n[bold cyan]Enter audio file/folder path (or number from list)[/bold cyan]").strip().strip("\"'")
    if raw.isdigit() and 1 <= int(raw) <= len(local_files):
        return local_files[int(raw) - 1]

    candidate = Path(raw).expanduser()
    if not candidate.exists():
        console.print(f"[bold red]Error:[/bold red] Path '{candidate}' does not exist.")
        sys.exit(1)
    return candidate


def _collect_audio_files(target: Path) -> list[Path]:
    """Resolve a single audio file or collect supported audio files from a directory."""
    if target.is_file():
        return [target]
    return sorted(p for p in target.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS)


def _render_analysis(a: TrackAnalysis) -> None:
    """Display a formatted Rich panel with technical metadata, loudness, and musical analysis."""
    grid = Table.grid(expand=True, padding=(0, 3))
    grid.add_column(ratio=1)
    grid.add_column(ratio=1)

    tech = Table(title="Technical & Signal Metrics", show_header=False, box=None, pad_edge=False)
    tech.add_column(style="cyan", width=16)
    tech.add_column(style="white")
    for label, val in (
        ("File", f"[bold]{a.file_name}[/bold] ({a.file_size_mb} MB)"),
        ("Duration", f"{a.duration_formatted} ({a.duration_seconds}s)"),
        ("Format", a.codec),
        ("Stream", f"{a.sample_rate_hz:,} Hz | {a.bitrate_kbps} kbps | {a.channel_layout}"),
        ("Peak / RMS", f"{a.peak_dbfs:+.2f} dBFS  /  {a.rms_dbfs:+.2f} dBFS"),
        ("Crest Factor", f"{a.crest_factor_db:.2f} dB"),
    ):
        tech.add_row(label, val)

    music = Table(title="Musical & DJ Analysis", show_header=False, box=None, pad_edge=False)
    music.add_column(style="magenta", width=16)
    music.add_column(style="white")
    music.add_row("Tempo (BPM)", f"[bold green]{a.bpm:.1f} BPM[/bold green]")
    music.add_row("Musical Key", f"[bold yellow]{a.musical_key}[/bold yellow] ({int(a.key_confidence * 100)}%)")
    music.add_row("Camelot Wheel", f"[bold bright_cyan]{a.camelot_key}[/bold bright_cyan]")
    for k, v in (a.tags.items() or [("ID3 Tags", "[dim]None[/dim]")]):
        music.add_row(k if k == "ID3 Tags" else f"Tag ({k})", v)

    grid.add_row(tech, music)
    console.print(Panel(grid, title=f"[bold bright_white]Track Analysis: {a.file_name}[/bold bright_white]", border_style="bright_blue"))


def _render_stems(stems: list[StemOutput], elapsed: float, track_dir: Path) -> None:
    """Display a summary table of generated MP3 stem files without path truncation."""
    table = Table(title=f"Separated Stems (320 kbps CBR MP3) — {elapsed:.1f}s", header_style="bold green")
    table.add_column("Stem", style="bold cyan", width=12)
    table.add_column("Size", justify="right", style="magenta", width=10)
    table.add_column("Output Path", style="white", overflow="fold")
    for s in stems:
        table.add_row(s.stem_name.upper(), f"{s.file_size_mb:.2f} MB", str(s.file_path))
    console.print(table)
    console.print(f"[dim]Output folder:[/dim] [bold]{track_dir}[/bold]\n")


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("input_path", required=False, type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output-dir", type=click.Path(path_type=Path), default=Path("output"), show_default=True, help="Output folder for stems.")
@click.option("-6", "--six-stems", is_flag=True, help="Split into 6 stems (vocals, drums, bass, guitar, piano, other).")
@click.option("-v", "--vocals", is_flag=True, help="Split only into vocals.mp3 + no_vocals.mp3 (acapella & instrumental).")
@click.option("-s", "--stem", "--two-stems", "two_stems", type=click.Choice(STEM_CHOICES, case_sensitive=False), help="Isolate a single <stem>.mp3 + no_<stem>.mp3.")
@click.option("-q", "--hq", is_flag=True, help="Use fine-tuned model (htdemucs_ft) for maximum separation quality.")
@click.option("-a", "--analyze", "--analyze-only", "analyze_only", is_flag=True, help="Analyze BPM, Key, and Loudness only (skip stem split).")
@click.option("-j", "--json", "--save-json", "save_json", is_flag=True, help="Save analysis.json alongside the stems.")
@click.option("-m", "--model", "model_name", type=click.Choice(list(SUPPORTED_MODELS), case_sensitive=False), default="htdemucs", show_default=True, help="Demucs v4 model.")
@click.option("-b", "--bitrate", type=click.Choice(["192", "256", "320"]), default="320", show_default=True, help="MP3 CBR bitrate (kbps).")
@click.option("--shifts", type=click.IntRange(0, 10), default=1, show_default=True, help="Time-shifts for stabilization.")
@click.option("--overlap", type=click.FloatRange(0.05, 0.9), default=0.25, show_default=True, help="Segment overlap ratio.")
@click.option("-d", "--device", type=click.Choice(["auto", "cpu", "cuda"], case_sensitive=False), default="auto", show_default=True, help="Compute device.")
@click.option("-t", "--threads", "--jobs", "jobs", type=int, default=0, show_default=True, help="CPU worker threads (0 = auto).")
@click.option("--download-models", is_flag=True, help="Pre-download local weights for models into ./models for 100% offline use.")
def cli(
    input_path: Path | None,
    output_dir: Path,
    six_stems: bool,
    vocals: bool,
    two_stems: str | None,
    hq: bool,
    analyze_only: bool,
    save_json: bool,
    model_name: str,
    bitrate: str,
    shifts: int,
    overlap: float,
    device: str,
    jobs: int,
    download_models: bool,
) -> None:
    """Analyze audio tracks (BPM, Key, Loudness) and split into 320 kbps MP3 stems (100% Local).

    Quick Examples:

    \b
      stems song.mp3             # 4 stems (vocals, drums, bass, other)
      stems song.mp3 -6          # 6 stems (+ guitar, piano)
      stems song.mp3 -v          # Vocals + Instrumental only
      stems song.mp3 -a          # Analyze BPM, Key & Loudness only
      stems song.mp3 -q          # Highest-quality fine-tuned model
      stems                      # Interactive prompt / current folder picker
    """
    if download_models:
        for m in ("htdemucs", "htdemucs_6s"):
            with console.status(f"[bold cyan]Caching local weights for '{m}' into {MODELS_DIR}...[/bold cyan]"):
                files = ensure_local_model(m)
            console.print(f"[green]✓[/green] {m}: {len(files)} weight file(s) ready in [bold]{MODELS_DIR}[/bold]")
        return

    target = _resolve_input_path(input_path)
    files = _collect_audio_files(target)
    if not files:
        console.print(f"[bold red]Error:[/bold red] No supported audio files found at '{target}'.")
        sys.exit(1)

    if six_stems:
        model_name = "htdemucs_6s"
    elif hq:
        model_name = "htdemucs_ft"

    if vocals and not two_stems:
        two_stems = "vocals"

    resolved_device = None if device.lower() == "auto" else device.lower()
    mp3_bitrate = int(bitrate)

    console.print(
        Panel.fit(
            f"[bold cyan]Stem Splitter[/bold cyan] • {model_name} (Local Offline Weights) • {mp3_bitrate} kbps CBR MP3",
            border_style="cyan",
        )
    )

    for idx, audio_file in enumerate(files, 1):
        if len(files) > 1:
            console.rule(f"[bold]Track {idx}/{len(files)}: {audio_file.name}[/bold]")

        with console.status(f"[bold cyan]Analyzing '{audio_file.name}'...[/bold cyan]", spinner="dots"):
            try:
                analysis = analyze_audio(audio_file)
            except Exception as exc:
                console.print(f"[bold red]Analysis failed for {audio_file.name}:[/bold red] {exc}")
                continue

        _render_analysis(analysis)
        track_out_dir = output_dir.resolve() / audio_file.stem

        if save_json:
            track_out_dir.mkdir(parents=True, exist_ok=True)
            json_path = track_out_dir / "analysis.json"
            json_path.write_text(json.dumps(analysis.to_dict(), indent=2), encoding="utf-8")
            console.print(f"[dim]Saved analysis JSON: {json_path}[/dim]")

        if analyze_only:
            continue

        start_t = time.perf_counter()
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(bar_width=36), TaskProgressColumn(), TimeElapsedColumn(), console=console) as prog:
            task = prog.add_task(f"[bold green]Splitting '{audio_file.name}' ({model_name})...[/bold green]", total=100)

            def _on_progress(info: dict) -> None:
                if info.get("state") == "end":
                    pct = ((info.get("model_idx_in_bag", 0) + min(1.0, info.get("segment_offset", 0) / max(1, info.get("audio_length", 1)))) / max(1, info.get("models", 1))) * 100.0
                    prog.update(task, completed=min(99.0, pct))

            try:
                stems = separate_to_stems(
                    audio_path=audio_file,
                    output_dir=output_dir,
                    model_name=model_name,
                    shifts=shifts,
                    overlap=overlap,
                    two_stems=two_stems,
                    mp3_bitrate=mp3_bitrate,
                    device=resolved_device,
                    jobs=jobs,
                    analysis=analysis,
                    progress_callback=_on_progress,
                )
                prog.update(task, completed=100.0, description="[bold green]Encoded 320 kbps MP3 stems![/bold green]")
            except Exception as exc:
                console.print(f"[bold red]Separation failed for {audio_file.name}:[/bold red] {exc}")
                continue

        _render_stems(stems, time.perf_counter() - start_t, track_out_dir)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
