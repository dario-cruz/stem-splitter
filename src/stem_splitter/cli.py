"""Rich CLI interface for Stem Splitter: Audio Analysis & Demucs v4 Stem Separation."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from .analyzer import TrackAnalysis, analyze_audio
from .separator import SUPPORTED_MODELS, StemOutput, separate_to_stems

console = Console()

AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac"}


def _collect_audio_files(target: Path) -> list[Path]:
    """Resolve a single audio file or recursively collect audio files from a directory."""
    if target.is_file():
        return [target]
    if target.is_dir():
        files = sorted(
            p for p in target.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
        )
        return files
    return []


def _render_analysis_table(analysis: TrackAnalysis) -> None:
    """Display a formatted Rich panel with technical metadata, loudness, and musical analysis."""
    grid = Table.grid(expand=True, padding=(0, 2))
    grid.add_column(justify="left", ratio=1)
    grid.add_column(justify="left", ratio=1)

    # Left sub-table: Technical & Signal Stats
    tech = Table(title="Technical & Signal Metrics", show_header=False, box=None, pad_edge=False)
    tech.add_column("Metric", style="cyan", width=18)
    tech.add_column("Value", style="white")
    tech.add_row("File", f"[bold]{analysis.file_name}[/bold] ({analysis.file_size_mb} MB)")
    tech.add_row("Duration", f"{analysis.duration_formatted} ({analysis.duration_seconds}s)")
    tech.add_row("Format / Codec", analysis.codec)
    tech.add_row(
        "Audio Stream",
        f"{analysis.sample_rate_hz:,} Hz | {analysis. bitrate_kbps} kbps | {analysis.channel_layout}",
    )
    tech.add_row("Peak Level", f"{analysis.peak_dbfs:+.2f} dBFS")
    tech.add_row("RMS Loudness", f"{analysis.rms_dbfs:+.2f} dBFS")
    tech.add_row("Crest Factor", f"{analysis.crest_factor_db:.2f} dB")

    # Right sub-table: Musical / DJ Analysis & ID3 Tags
    music = Table(title="Musical & DJ Analysis", show_header=False, box=None, pad_edge=False)
    music.add_column("Property", style="magenta", width=18)
    music.add_column("Value", style="white")
    music.add_row("Tempo (BPM)", f"[bold green]{analysis.bpm:.1f} BPM[/bold green]")
    music.add_row(
        "Musical Key",
        f"[bold yellow]{analysis.musical_key}[/bold yellow] (Confidence: {int(analysis.key_confidence * 100)}%)",
    )
    music.add_row("Camelot Wheel", f"[bold bright_cyan]{analysis.camelot_key}[/bold bright_cyan]")

    if analysis.tags:
        for k, v in analysis.tags.items():
            music.add_row(f"Tag ({k})", v)
    else:
        music.add_row("ID3 Tags", "[dim]None present[/dim]")

    grid.add_row(tech, music)
    console.print(
        Panel(
            grid,
            title=f"[bold bright_white]Track Analysis: {analysis.file_name}[/bold bright_white]",
            border_style="bright_blue",
        )
    )


def _render_stems_summary(stems: list[StemOutput], elapsed_sec: float, track_dir: Path) -> None:
    """Display a summary table of generated MP3 stem files."""
    table = Table(
        title=f"Separated Stems (320 kbps CBR MP3, LAME q=2) — Completed in {elapsed_sec:.1f}s",
        show_lines=False,
        header_style="bold green",
    )
    table.add_column("Stem", style="bold cyan", width=14)
    table.add_column("Size", justify="right", style="magenta", width=10)
    table.add_column("Output Path", style="white")

    for stem in stems:
        table.add_row(stem.stem_name.upper(), f"{stem.file_size_mb:.2f} MB", str(stem.file_path))

    console.print(table)
    console.print(f"[dim]Output folder:[/dim] [bold]{track_dir}[/bold]\n")


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("output"),
    show_default=True,
    help="Base directory where separated MP3 stems will be saved.",
)
@click.option(
    "-m",
    "--model",
    "model_name",
    type=click.Choice(list(SUPPORTED_MODELS.keys()), case_sensitive=False),
    default="htdemucs",
    show_default=True,
    help="Demucs v4 model: htdemucs (4 stems), htdemucs_ft (4 stems, fine-tuned), htdemucs_6s (6 stems: +guitar, +piano).",
)
@click.option(
    "--six-stems",
    is_flag=True,
    help="Shortcut for --model htdemucs_6s (separates vocals, drums, bass, guitar, piano, other).",
)
@click.option(
    "--two-stems",
    type=click.Choice(["vocals", "drums", "bass", "other", "guitar", "piano"], case_sensitive=False),
    default=None,
    help="Only split into <STEM>.mp3 and no_<STEM>.mp3 (e.g., --two-stems vocals for acapella + instrumental).",
)
@click.option(
    "--shifts",
    type=click.IntRange(0, 10),
    default=1,
    show_default=True,
    help="Number of random time-shifts for equivariant stabilization (higher = better quality, slower).",
)
@click.option(
    "--overlap",
    type=click.FloatRange(0.05, 0.9),
    default=0.25,
    show_default=True,
    help="Overlap ratio between split audio segments.",
)
@click.option(
    "-b",
    "--bitrate",
    type=click.Choice(["192", "256", "320"]),
    default="320",
    show_default=True,
    help="Output MP3 constant bitrate (kbps).",
)
@click.option(
    "--analyze-only",
    is_flag=True,
    help="Only run technical and musical (BPM/Key) analysis without separating stems.",
)
@click.option(
    "--save-json",
    is_flag=True,
    help="Save an analysis.json file alongside the separated MP3 stems.",
)
@click.option(
    "-d",
    "--device",
    type=click.Choice(["auto", "cpu", "cuda"], case_sensitive=False),
    default="auto",
    show_default=True,
    help="Compute device for Demucs inference.",
)
@click.option(
    "-j",
    "--jobs",
    type=int,
    default=0,
    show_default=True,
    help="Number of parallel CPU threads (0 = auto).",
)
def cli(
    input_path: Path,
    output_dir: Path,
    model_name: str,
    six_stems: bool,
    two_stems: str | None,
    shifts: int,
    overlap: float,
    bitrate: str,
    analyze_only: bool,
    save_json: bool,
    device: str,
    jobs: int,
) -> None:
    """Analyze an MP3/audio file (or folder of files) and separate it into highest-quality MP3 stems.

    Examples:

      stem-splitter track.mp3

      stem-splitter track.mp3 --six-stems

      stem-splitter track.mp3 --two-stems vocals -o ./stems

      stem-splitter ./my_tracks_folder --analyze-only
    """
    if six_stems:
        model_name = "htdemucs_6s"

    files = _collect_audio_files(input_path)
    if not files:
        console.print(f"[bold red]Error:[/bold red] No supported audio files found at '{input_path}'.")
        sys.exit(1)

    resolved_device = None if device.lower() == "auto" else device.lower()
    mp3_bitrate = int(bitrate)

    console.print(
        Panel.fit(
            f"[bold cyan]Stem Splitter[/bold cyan] • Demucs v4 ({model_name}) • {mp3_bitrate} kbps CBR MP3 (LAME Preset 2)",
            border_style="cyan",
        )
    )

    for idx, audio_file in enumerate(files, start=1):
        if len(files) > 1:
            console.rule(f"[bold]Track {idx} of {len(files)}: {audio_file.name}[/bold]")

        # Step 1: Audio Analysis
        with console.status(
            f"[bold cyan]Analyzing audio signal, loudness, BPM, and musical key for '{audio_file.name}'...[/bold cyan]",
            spinner="dots",
        ):
            try:
                analysis = analyze_audio(audio_file)
            except Exception as exc:
                console.print(f"[bold red]Analysis failed for {audio_file.name}:[/bold red] {exc}")
                continue

        _render_analysis_table(analysis)

        track_out_dir = output_dir.resolve() / audio_file.stem
        if save_json:
            track_out_dir.mkdir(parents=True, exist_ok=True)
            json_path = track_out_dir / "analysis.json"
            json_path.write_text(json.dumps(analysis.to_dict(), indent=2), encoding="utf-8")
            console.print(f"[dim]Saved analysis metadata to {json_path}[/dim]")

        if analyze_only:
            continue

        # Step 2: Stem Separation with Rich progress bar
        start_time = time.perf_counter()
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            sep_task = progress.add_task(
                f"[bold green]Separating '{audio_file.name}' with {model_name}...[/bold green]",
                total=100,
            )

            def _on_demucs_progress(info: dict) -> None:
                if info.get("state") == "end":
                    audio_len = max(1, info.get("audio_length", 1))
                    offset = info.get("segment_offset", 0)
                    models_count = max(1, info.get("models", 1))
                    model_idx = info.get("model_idx_in_bag", 0)
                    # Approximate overall completion percentage across submodels and segments
                    seg_frac = min(1.0, offset / audio_len)
                    overall_pct = ((model_idx + seg_frac) / models_count) * 100.0
                    progress.update(sep_task, completed=min(99.0, overall_pct))

            try:
                stems = separate_to_stems(
                    audio_path=audio_file,
                    output_dir=output_dir,
                    model_name=model_name,
                    shifts=shifts,
                    overlap=overlap,
                    two_stems=two_stems,
                    mp3_bitrate=mp3_bitrate,
                    mp3_preset=2,
                    device=resolved_device,
                    jobs=jobs,
                    analysis=analysis,
                    progress_callback=_on_demucs_progress,
                )
                progress.update(sep_task, completed=100.0, description="[bold green]Encoded 320kbps MP3 stems![/bold green]")
            except Exception as exc:
                console.print(f"[bold red]Stem separation failed for {audio_file.name}:[/bold red] {exc}")
                continue

        elapsed = time.perf_counter() - start_time
        _render_stems_summary(stems, elapsed, track_out_dir)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
