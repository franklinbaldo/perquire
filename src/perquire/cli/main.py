"""Small supported CLI for Perquire."""

from __future__ import annotations

import csv
import json
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Annotated, Optional

import numpy as np
import typer
from rich.console import Console
from rich.prompt import Confirm
from rich.table import Table

from ..providers import list_available_providers

console = Console()
app = typer.Typer(
    name="perquire",
    help="Perquire: experimental semantic inversion in a known embedding space.",
    add_completion=False,
    rich_markup_mode="rich",
)


def _version() -> str:
    try:
        return package_version("perquire")
    except PackageNotFoundError:
        return "0.2.0"


def version_callback(value: bool) -> None:
    if value:
        console.print(f"Perquire {_version()}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        Optional[bool],
        typer.Option("--version", "-v", callback=version_callback, is_eager=True),
    ] = None,
) -> None:
    """Investigate target embeddings using similarity-guided text candidates."""


@app.command()
def providers() -> None:
    """List provider integrations and whether their dependencies are installed."""
    data = list_available_providers()
    for kind, title in (("embedding", "Embedding Providers"), ("llm", "LLM Providers")):
        table = Table(title=title)
        table.add_column("Provider")
        table.add_column("Status")
        table.add_column("Extra")
        for name, info in data[kind].items():
            table.add_row(
                name,
                "installed" if info["installed"] else "not installed",
                str(info.get("extra", "-")),
            )
        console.print(table)


@app.command()
def configure(
    provider: Annotated[Optional[str], typer.Option("--provider", "-p")] = None,
    api_key: Annotated[Optional[str], typer.Option("--api-key", "-k")] = None,
    database: Annotated[Optional[str], typer.Option("--database", "-d")] = None,
    show: Annotated[bool, typer.Option("--show")] = False,
) -> None:
    """Store small local defaults for provider selection and persistence."""
    config_file = Path.home() / ".perquire" / "config.json"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config = get_global_config()

    if show:
        table = Table(title="Current Configuration")
        table.add_column("Setting")
        table.add_column("Value")
        for key, value in sorted(config.items()):
            rendered = "***" if "key" in key.lower() and value else str(value)
            table.add_row(key, rendered)
        console.print(table)
        return

    if provider:
        config["default_provider"] = provider
        config["default_llm_provider"] = provider
        config["default_embedding_provider"] = provider
    if api_key:
        selected = provider or config.get("default_provider", "gemini")
        config[f"{selected}_api_key"] = api_key
    if database:
        config["default_database"] = database

    config_file.write_text(json.dumps(config, indent=2), encoding="utf-8")
    console.print(f"Configuration saved to: {config_file}")


def _open_database(path: str):
    from ..database.base import DatabaseConfig
    from ..database.duckdb_provider import DuckDBProvider

    database = DuckDBProvider(DatabaseConfig(connection_string=path))
    database.connect()
    return database


@app.command()
def status(
    database: Annotated[str, typer.Option("--database", "-d")] = "perquire.db",
) -> None:
    """Show persisted investigation statistics and recent results."""
    db = _open_database(database)
    try:
        stats = db.get_investigation_stats()
        table = Table(title="Database Statistics")
        table.add_column("Metric")
        table.add_column("Value")
        table.add_row("Total Investigations", str(stats.get("total_investigations", 0)))
        table.add_row("Total Questions", str(stats.get("total_questions", 0)))
        table.add_row("Average Similarity", f"{stats.get('average_similarity', 0):.3f}")
        table.add_row("Average Iterations", f"{stats.get('average_iterations', 0):.1f}")
        console.print(table)

        recent = db.list_investigations(limit=5)
        if recent:
            recent_table = Table(title="Recent Investigations")
            recent_table.add_column("ID")
            recent_table.add_column("Description")
            recent_table.add_column("Similarity")
            recent_table.add_column("Strategy")
            for item in recent:
                recent_table.add_row(
                    str(item["investigation_id"]),
                    str(item["description"]),
                    f"{item['final_similarity']:.3f}",
                    str(item["strategy_name"]),
                )
            console.print(recent_table)
    finally:
        db.disconnect()


@app.command()
def export(
    database: Annotated[str, typer.Option("--database", "-d")] = "perquire.db",
    output: Annotated[str, typer.Option("--output", "-o")] = "investigations.json",
    format: Annotated[str, typer.Option("--format", "-f")] = "json",
    limit: Annotated[Optional[int], typer.Option("--limit", "-l")] = None,
) -> None:
    """Export persisted investigation summaries as JSON, CSV, or text."""
    if format not in {"json", "csv", "txt"}:
        raise typer.BadParameter("format must be json, csv, or txt")

    db = _open_database(database)
    try:
        investigations = db.list_investigations(limit=limit or 100)
    finally:
        db.disconnect()

    output_path = Path(output)
    if format == "json":
        output_path.write_text(
            json.dumps(investigations, indent=2, default=str), encoding="utf-8"
        )
    elif format == "csv":
        fieldnames = list(investigations[0].keys()) if investigations else []
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if fieldnames:
                writer.writeheader()
                writer.writerows(investigations)
    else:
        with output_path.open("w", encoding="utf-8") as handle:
            for item in investigations:
                handle.write(f"ID: {item['investigation_id']}\n")
                handle.write(f"Description: {item['description']}\n")
                handle.write(f"Similarity: {item['final_similarity']}\n")
                handle.write(f"Strategy: {item['strategy_name']}\n")
                handle.write("-" * 50 + "\n")

    console.print(f"Exported {len(investigations)} investigations to {output_path}")


def get_global_config() -> dict:
    config_file = Path.home() / ".perquire" / "config.json"
    if not config_file.exists():
        return {}
    try:
        return json.loads(config_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def load_embedding_from_file(file_path: Path, format_type: str) -> np.ndarray:
    if format_type == "npy":
        return np.load(file_path)
    if format_type == "json":
        return np.asarray(json.loads(file_path.read_text(encoding="utf-8")), dtype=float)
    if format_type == "txt":
        return np.loadtxt(file_path)
    raise ValueError(f"Unsupported format: {format_type}")


def list_embedding_files(directory: Path, format_type: str, limit: Optional[int] = None) -> list[Path]:
    patterns = {"npy": "*.npy", "json": "*.json", "txt": "*.txt"}
    if format_type not in patterns:
        raise ValueError(f"Unsupported format: {format_type}")
    files = sorted(directory.glob(patterns[format_type]))
    return files[:limit] if limit is not None else files


def create_investigator_from_cli_options(
    llm_provider_name: Optional[str],
    embedding_provider_name: Optional[str],
    strategy_name: Optional[str],
    database_path_cli: Optional[str],
    verbose: bool = False,
):
    from ..core.investigator import PerquireInvestigator
    from ..embeddings import embedding_registry
    from ..llm import provider_registry as llm_registry

    config = get_global_config()
    llm_name = llm_provider_name or config.get("default_llm_provider")
    embedding_name = embedding_provider_name or config.get("default_embedding_provider")

    if not llm_name:
        available = llm_registry.list_providers()
        if not available:
            raise typer.BadParameter("No LLM provider is available")
        llm_name = available[0]
    if not embedding_name:
        available = embedding_registry.list_providers()
        if not available:
            raise typer.BadParameter("No embedding provider is available")
        embedding_name = available[0]

    db_path = database_path_cli or config.get("default_database")
    db = _open_database(str(db_path)) if db_path else None
    return PerquireInvestigator(
        llm_provider=llm_name,
        embedding_provider=embedding_name,
        database_provider=db,
    )


def display_investigation_result(result, verbose: bool = False) -> None:
    console.print("Investigation Complete")
    console.print(f"Description: {result.description}")
    console.print(f"Similarity: {result.final_similarity:.4f}")
    console.print(f"Iterations: {result.iterations}")
    if verbose and result.question_history:
        table = Table(title="Probe History")
        table.add_column("Step")
        table.add_column("Phase")
        table.add_column("Candidate")
        table.add_column("Similarity")
        for index, item in enumerate(result.question_history, start=1):
            table.add_row(
                str(index), item.phase, item.question, f"{item.similarity_score:.4f}"
            )
        console.print(table)


@app.command()
def investigate(
    embedding_file: Annotated[Path, typer.Argument(exists=True)],
    llm_provider: Annotated[Optional[str], typer.Option("--llm-provider")] = None,
    embedding_provider: Annotated[Optional[str], typer.Option("--embedding-provider")] = None,
    database: Annotated[Optional[str], typer.Option("--database", "-d")] = None,
    file_format: Annotated[str, typer.Option("--format", "-f")] = "npy",
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Investigate one target embedding."""
    investigator = create_investigator_from_cli_options(
        llm_provider, embedding_provider, None, database, verbose
    )
    target = load_embedding_from_file(embedding_file, file_format)
    result = investigator.investigate(target_embedding=target, verbose=verbose)
    display_investigation_result(result, verbose)


@app.command()
def batch(
    directory: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    llm_provider: Annotated[Optional[str], typer.Option("--llm-provider")] = None,
    embedding_provider: Annotated[Optional[str], typer.Option("--embedding-provider")] = None,
    database: Annotated[Optional[str], typer.Option("--database", "-d")] = None,
    file_format: Annotated[str, typer.Option("--format", "-f")] = "npy",
    limit: Annotated[Optional[int], typer.Option("--limit", "-l")] = None,
    yes: Annotated[bool, typer.Option("--yes", "-y")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Investigate every supported embedding file in a directory."""
    files = list_embedding_files(directory, file_format, limit)
    if not files:
        console.print("No embedding files found")
        return
    if not yes and not Confirm.ask(f"Investigate {len(files)} files?"):
        raise typer.Abort()

    investigator = create_investigator_from_cli_options(
        llm_provider, embedding_provider, None, database, verbose
    )
    rows: list[tuple[Path, object | None]] = []
    for file_path in files:
        try:
            target = load_embedding_from_file(file_path, file_format)
            rows.append((file_path, investigator.investigate(target, verbose=verbose)))
        except Exception as exc:  # continue batch while making the failure visible
            console.print(f"Failed {file_path.name}: {exc}")
            rows.append((file_path, None))

    table = Table(title=f"Batch Investigation Summary ({len(rows)} files)")
    table.add_column("File")
    table.add_column("Description")
    table.add_column("Similarity")
    for file_path, result in rows:
        if result is None:
            table.add_row(file_path.name, "Failed", "-")
        else:
            table.add_row(file_path.name, result.description, f"{result.final_similarity:.4f}")
    console.print(table)


@app.command()
def serve(
    host: Annotated[str, typer.Option("--host")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port")] = 8000,
    database: Annotated[str, typer.Option("--database")] = "perquire.db",
    reload: Annotated[bool, typer.Option("--reload")] = False,
) -> None:
    """Launch the optional local FastAPI/Jinja interface."""
    try:
        from ..web.main import main as web_main
    except ImportError as exc:
        console.print(f"Web dependencies are not installed: {exc}")
        raise typer.Exit(1) from exc

    import sys

    original = sys.argv
    sys.argv = ["perquire-web", "--host", host, "--port", str(port), "--database", database]
    if reload:
        sys.argv.append("--reload")
    try:
        web_main()
    finally:
        sys.argv = original


if __name__ == "__main__":
    app()
