"""
Main CLI interface for Perquire using Typer.
"""

import typer
from typing import Optional, List, Annotated
from pathlib import Path
import json
import numpy as np

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.prompt import Confirm
from rich.panel import Panel

from ..providers import list_available_providers, ProviderNotInstalledError

console = Console()
app = typer.Typer(
    name="perquire",
    help="🔍 Perquire: Reverse Embedding Search Through Systematic Questioning",
    add_completion=False,
    rich_markup_mode="rich",
)


def version_callback(value: bool):
    """Show version and exit."""
    if value:
        console.print("[bold cyan]Perquire[/bold cyan] version [green]0.2.0[/green]")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        Optional[bool],
        typer.Option("--version", "-v", callback=version_callback, is_eager=True, help="Show version and exit"),
    ] = None,
):
    """
    [bold cyan]Perquire[/bold cyan]: Investigate mysterious embeddings through systematic questioning.

    A revolutionary AI system that reverses traditional embedding search.
    """
    pass


@app.command()
def providers():
    """
    📋 List available LLM and embedding providers and their installation status.
    """
    try:
        providers_data = list_available_providers()

        console.print("\n[bold]📋 Available Providers[/bold]\n")

        # Embedding providers
        console.print("[bold blue]🔍 Embedding Providers[/bold blue]")
        embed_table = Table(show_header=True, header_style="bold magenta")
        embed_table.add_column("Provider", style="cyan")
        embed_table.add_column("Status", justify="center")
        embed_table.add_column("Install Command", style="dim")

        for name, info in providers_data["embedding"].items():
            status = "✅ Installed" if info["installed"] else "❌ Not installed"
            install_cmd = f"uv add perquire[{info['extra']}]" if not info["installed"] else "-"
            embed_table.add_row(name, status, install_cmd)

        console.print(embed_table)
        console.print()

        # LLM providers
        console.print("[bold blue]🤖 LLM Providers[/bold blue]")
        llm_table = Table(show_header=True, header_style="bold magenta")
        llm_table.add_column("Provider", style="cyan")
        llm_table.add_column("Status", justify="center")
        llm_table.add_column("Install Command", style="dim")

        for name, info in providers_data["llm"].items():
            status = "✅ Installed" if info["installed"] else "❌ Not installed"
            install_cmd = f"uv add perquire[{info['extra']}]" if not info["installed"] else "-"
            llm_table.add_row(name, status, install_cmd)

        console.print(llm_table)

        # Installation examples
        console.print("\n[bold yellow]💡 Common Installation Examples:[/bold yellow]")
        console.print("   uv add perquire[api-openai,dev]     # OpenAI + dev tools")
        console.print("   uv add perquire[api-gemini,web]     # Gemini + web interface")
        console.print("   uv add perquire[local-embeddings]   # Local inference (heavy!)")
        console.print("   uv add perquire[api-openai,api-anthropic,web,dev]  # Full setup")

    except Exception as e:
        console.print(f"[red]Error listing providers: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def configure(
    provider: Annotated[
        Optional[str],
        typer.Option("--provider", "-p", help="Set default LLM provider (gemini, openai, anthropic, ollama)"),
    ] = None,
    api_key: Annotated[Optional[str], typer.Option("--api-key", "-k", help="Set API key for provider")] = None,
    database: Annotated[Optional[str], typer.Option("--database", "-d", help="Set default database path")] = None,
    show: Annotated[bool, typer.Option("--show", help="Show current configuration")] = False,
):
    """
    ⚙️  Configure Perquire settings.
    """
    config_file = Path.home() / '.perquire' / 'config.json'
    config_file.parent.mkdir(exist_ok=True)

    # Load existing config
    config = {}
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)

    # Show current config
    if show:
        console.print("[bold]Current Configuration:[/bold]")
        table = Table()
        table.add_column("Setting")
        table.add_column("Value")

        for key, value in config.items():
            if 'key' in key.lower():
                value = "***" if value else "[red]Not set[/red]"
            table.add_row(key, str(value))

        console.print(table)
        return

    # Update config
    if provider:
        config['default_provider'] = provider
        console.print(f"✅ [green]Set default provider to:[/green] {provider}")

    if api_key:
        key_name = f"{config.get('default_provider', 'gemini')}_api_key"
        config[key_name] = api_key
        console.print(f"✅ [green]Set API key for:[/green] {config.get('default_provider', 'gemini')}")

    if database:
        config['default_database'] = database
        console.print(f"✅ [green]Set default database to:[/green] {database}")

    # Save config
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    console.print(f"💾 [green]Configuration saved to:[/green] {config_file}")


@app.command()
def status(
    database: Annotated[str, typer.Option("--database", "-d", help="Database file path")] = "perquire.db",
):
    """
    📊 Show investigation status and statistics.
    """
    try:
        from ..database.duckdb_provider import DuckDBProvider
        from ..database.base import DatabaseConfig

        # Connect to database
        db_config = DatabaseConfig(connection_string=database)
        db_provider = DuckDBProvider(db_config)

        # Get statistics
        stats = db_provider.get_statistics()

        console.print("[bold]🔍 Perquire Investigation Status[/bold]")
        console.print()

        # Basic stats table
        stats_table = Table(title="Database Statistics")
        stats_table.add_column("Metric")
        stats_table.add_column("Value")

        stats_table.add_row("Total Investigations", str(stats.get('total_investigations', 0)))
        stats_table.add_row("Total Questions", str(stats.get('total_questions', 0)))
        stats_table.add_row("Average Similarity", f"{stats.get('avg_similarity', 0):.3f}")
        stats_table.add_row("Average Iterations", f"{stats.get('avg_iterations', 0):.1f}")

        console.print(stats_table)

        # Recent investigations
        recent = db_provider.get_recent_investigations(limit=5)
        if recent:
            console.print()
            recent_table = Table(title="Recent Investigations")
            recent_table.add_column("ID")
            recent_table.add_column("Description")
            recent_table.add_column("Similarity")
            recent_table.add_column("Strategy")

            for inv in recent:
                recent_table.add_row(
                    inv['investigation_id'][:8] + "...",
                    inv['description'][:50] + "..." if len(inv['description']) > 50 else inv['description'],
                    f"{inv['final_similarity']:.3f}",
                    inv['strategy_name']
                )

            console.print(recent_table)

    except Exception as e:
        console.print(f"❌ [red]Failed to get status:[/red] {str(e)}")
        raise typer.Exit(1)


@app.command()
def export(
    database: Annotated[str, typer.Option("--database", "-d", help="Database file path")] = "perquire.db",
    output: Annotated[str, typer.Option("--output", "-o", help="Output file path")] = "investigations.json",
    format: Annotated[str, typer.Option("--format", "-f", help="Export format")] = "json",
    limit: Annotated[Optional[int], typer.Option("--limit", "-l", help="Limit number of records")] = None,
):
    """
    📤 Export investigation results.
    """
    if format not in ['json', 'csv', 'txt']:
        console.print(f"[red]Invalid format: {format}. Must be json, csv, or txt[/red]")
        raise typer.Exit(1)

    try:
        from ..database.duckdb_provider import DuckDBProvider
        from ..database.base import DatabaseConfig

        # Connect to database
        db_config = DatabaseConfig(connection_string=database)
        db_provider = DuckDBProvider(db_config)

        # Get investigations
        investigations = db_provider.get_all_investigations(limit=limit)

        if not investigations:
            console.print("❌ [red]No investigations found in database[/red]")
            return

        # Export based on format
        output_path = Path(output)

        if format == 'json':
            with open(output_path, 'w') as f:
                json.dump(investigations, f, indent=2, default=str)

        elif format == 'csv':
            import pandas as pd
            df = pd.DataFrame(investigations)
            df.to_csv(output_path, index=False)

        elif format == 'txt':
            with open(output_path, 'w') as f:
                for inv in investigations:
                    f.write(f"ID: {inv['investigation_id']}\n")
                    f.write(f"Description: {inv['description']}\n")
                    f.write(f"Similarity: {inv['final_similarity']}\n")
                    f.write(f"Strategy: {inv['strategy_name']}\n")
                    f.write("-" * 50 + "\n")

        console.print(f"📤 [green]Exported {len(investigations)} investigations to:[/green] {output_path}")

    except Exception as e:
        console.print(f"❌ [red]Export failed:[/red] {str(e)}")
        raise typer.Exit(1)


# Helper functions
def get_global_config() -> dict:
    """Loads global configuration from ~/.perquire/config.json."""
    config_file = Path.home() / '.perquire' / 'config.json'
    config_data = {}
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
        except json.JSONDecodeError:
            console.print(f"[yellow]Warning: Could not parse config file at {config_file}[/yellow]")
    return config_data


def load_embedding_from_file(file_path: Path, format_type: str) -> np.ndarray:
    """Load embedding from file based on format."""
    if format_type == 'npy':
        return np.load(file_path)
    elif format_type == 'json':
        with open(file_path, 'r') as f:
            data = json.load(f)
            return np.array(data)
    elif format_type == 'txt':
        return np.loadtxt(file_path)
    else:
        raise ValueError(f"Unsupported format: {format_type}")


def list_embedding_files(directory: Path, format_type: str, limit: Optional[int] = None) -> List[Path]:
    """List embedding files in directory."""
    patterns = {'npy': '*.npy', 'json': '*.json', 'txt': '*.txt'}

    if format_type not in patterns:
        raise ValueError(f"Unsupported format: {format_type}")

    files = list(directory.glob(patterns[format_type]))

    if limit:
        files = files[:limit]

    return files


def create_investigator_from_cli_options(
    llm_provider_name: Optional[str],
    embedding_provider_name: Optional[str],
    strategy_name: Optional[str],
    database_path_cli: Optional[str],
    verbose: bool = False
):
    """Create PerquireInvestigator from CLI options."""
    from ..core.investigator import PerquireInvestigator
    from ..exceptions import ConfigurationError, InvestigationError
    from ..llm import provider_registry as llm_registry
    from ..embeddings import embedding_registry

    global_config = get_global_config()

    final_llm_provider_name = llm_provider_name or global_config.get("default_llm_provider")
    if not final_llm_provider_name:
        available_llms = llm_registry.list_providers()
        if available_llms:
            final_llm_provider_name = available_llms[0]
            if verbose:
                console.print(f"[dim]Using first available LLM: [cyan]{final_llm_provider_name}[/cyan][/dim]")
        else:
            raise typer.Exit("No LLM provider available. Configure one or ensure API keys are set.")

    final_embedding_provider_name = embedding_provider_name or global_config.get("default_embedding_provider")
    if not final_embedding_provider_name:
        available_embeddings = embedding_registry.list_providers()
        if available_embeddings:
            final_embedding_provider_name = available_embeddings[0]
            if verbose:
                console.print(f"[dim]Using first available embedding: [cyan]{final_embedding_provider_name}[/cyan][/dim]")
        else:
            raise typer.Exit("No embedding provider available. Configure one or ensure API keys are set.")

    db_path = database_path_cli or global_config.get('default_database')
    db_provider_instance = None
    if db_path:
        try:
            from ..database.duckdb_provider import DuckDBProvider
            from ..database.base import DatabaseConfig
            db_config = DatabaseConfig(connection_string=str(db_path))
            db_provider_instance = DuckDBProvider(db_config)
            if verbose:
                console.print(f"[dim]Using database: {Path(db_path).resolve()}[/dim]")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not initialize database: {e}[/yellow]")

    try:
        investigator = PerquireInvestigator(
            llm_provider=final_llm_provider_name,
            embedding_provider=final_embedding_provider_name,
            database_provider=db_provider_instance,
        )
        if verbose:
            console.print(f"[dim]Investigator: LLM=[cyan]{final_llm_provider_name}[/cyan], Embeddings=[cyan]{final_embedding_provider_name}[/cyan][/dim]")
        return investigator
    except (ConfigurationError, InvestigationError) as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


def display_investigation_result(result, verbose: bool = False):
    """Display investigation result."""
    from ..core.result import InvestigationResult

    if not isinstance(result, InvestigationResult):
        console.print(f"[yellow]Cannot display result: unexpected type {type(result)}[/yellow]")
        return

    console.print("\n[bold green]✅ Investigation Complete![/bold green]")
    console.print(f"   [bold]Description:[/bold] {result.description}")
    console.print(f"   [bold]Similarity:[/bold]  {result.final_similarity:.4f}")
    console.print(f"   [bold]Iterations:[/bold]  {result.iterations}")
    console.print(f"   [bold]Duration:[/bold]    {result.total_duration_seconds:.2f}s")
    console.print(f"   [bold]Strategy:[/bold]    {result.strategy_name}")

    if verbose and hasattr(result, 'question_history') and result.question_history:
        console.print("\n[bold]Question History:[/bold]")
        history_table = Table(show_header=True, header_style="bold magenta")
        history_table.add_column("Iter.", style="dim")
        history_table.add_column("Phase", style="cyan")
        history_table.add_column("Question", overflow="fold")
        history_table.add_column("Similarity", style="magenta")

        for i, qr in enumerate(result.question_history):
            history_table.add_row(str(i + 1), qr.phase, qr.question, f"{qr.similarity:.4f}")
        console.print(history_table)


@app.command()
def investigate(
    embedding_file: Annotated[Path, typer.Argument(help="Path to embedding file (.npy, .json, .txt)", exists=True)],
    llm_provider: Annotated[Optional[str], typer.Option("--llm-provider", help="LLM provider")] = None,
    embedding_provider: Annotated[Optional[str], typer.Option("--embedding-provider", help="Embedding provider")] = None,
    strategy: Annotated[Optional[str], typer.Option("--strategy", help="Questioning strategy")] = None,
    database: Annotated[Optional[str], typer.Option("--database", "-d", help="Database path")] = None,
    file_format: Annotated[str, typer.Option("--format", "-f", help="File format")] = "npy",
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable verbose output")] = False,
):
    """
    🔎 Investigate a single embedding file to uncover what it represents.
    """
    if verbose:
        console.print(f"[bold blue]🔎 Starting investigation: {embedding_file.name}[/bold blue]")
    else:
        console.print(f"🔎 Investigating {embedding_file.name}...")

    try:
        investigator = create_investigator_from_cli_options(
            llm_provider, embedding_provider, strategy, database, verbose=verbose
        )

        target_embedding = load_embedding_from_file(embedding_file, file_format)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as pb:
            task_id = pb.add_task("Investigating...", total=None)
            result = investigator.investigate(target_embedding=target_embedding, verbose=verbose)
            pb.update(task_id, completed=True, description="Investigation complete.")

        display_investigation_result(result, verbose)

    except Exception as e:
        console.print(f"❌ [red]Investigation Error:[/red] {str(e)}")
        if verbose:
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise typer.Exit(1)


@app.command()
def batch(
    embeddings_dir: Annotated[Path, typer.Argument(help="Directory containing embedding files", exists=True, file_okay=False)],
    llm_provider: Annotated[Optional[str], typer.Option("--llm-provider", help="LLM provider")] = None,
    embedding_provider: Annotated[Optional[str], typer.Option("--embedding-provider", help="Embedding provider")] = None,
    strategy: Annotated[Optional[str], typer.Option("--strategy", help="Questioning strategy")] = None,
    database: Annotated[Optional[str], typer.Option("--database", "-d", help="Database path")] = None,
    limit: Annotated[Optional[int], typer.Option("--limit", "-l", help="Limit number of files")] = None,
    file_format: Annotated[str, typer.Option("--format", "-f", help="File format")] = "npy",
    output_dir: Annotated[Optional[str], typer.Option("--output-dir", help="Output directory for results")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable verbose output")] = False,
):
    """
    🚀 Investigate multiple embedding files from a directory.
    """
    console.print(f"[bold blue]🚀 Starting batch investigation in: {embeddings_dir}[/bold blue]")

    try:
        investigator = create_investigator_from_cli_options(
            llm_provider, embedding_provider, strategy, database, verbose=False
        )

        embedding_files = list_embedding_files(embeddings_dir, file_format, limit)
        if not embedding_files:
            console.print(f"[yellow]No {file_format} files found in {embeddings_dir}[/yellow]")
            return

        console.print(f"Found {len(embedding_files)} files")
        if not Confirm.ask("Proceed with batch investigation?", default=True):
            console.print("[yellow]Cancelled[/yellow]")
            return

        batch_results = []
        output_path = Path(output_dir) if output_dir else None
        if output_path:
            output_path.mkdir(parents=True, exist_ok=True)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}% ({task.completed}/{task.total})"),
            console=console
        ) as progress:
            task_id = progress.add_task("Batch processing...", total=len(embedding_files))

            for file_p in embedding_files:
                progress.update(task_id, description=f"Processing: {file_p.name}")
                try:
                    emb = load_embedding_from_file(file_p, file_format)
                    res = investigator.investigate(target_embedding=emb, verbose=verbose)
                    batch_results.append((file_p, res))

                    if output_path and res and hasattr(res, 'to_dict'):
                        try:
                            with open(output_path / f"{file_p.stem}_result.json", 'w') as f:
                                json.dump(res.to_dict(), f, indent=2, default=str)
                        except Exception as e:
                            console.print(f"[red]Error saving JSON for {file_p.name}: {e}[/red]")

                    if verbose and res:
                        console.rule(f"Result for {file_p.name}")
                        display_investigation_result(res, verbose=True)
                        console.rule()

                except Exception as e:
                    console.print(f"[red]Failed {file_p.name}: {e}[/red]")
                    batch_results.append((file_p, None))
                finally:
                    progress.update(task_id, advance=1)

        # Display summary
        from ..core.result import InvestigationResult

        console.print(f"\n[bold green]📊 Batch Investigation Summary ({len(batch_results)} files)[/bold green]")
        summary_table = Table(title="Batch Results", show_header=True, header_style="bold magenta")
        summary_table.add_column("File", style="cyan", overflow="fold", max_width=50)
        summary_table.add_column("Description", overflow="fold")
        summary_table.add_column("Similarity", style="magenta", justify="right")
        summary_table.add_column("Iterations", style="blue", justify="right")

        successful = 0
        for file_p, res in batch_results:
            if isinstance(res, InvestigationResult):
                summary_table.add_row(
                    str(file_p.name),
                    res.description,
                    f"{res.final_similarity:.4f}",
                    str(res.iterations)
                )
                successful += 1
            else:
                summary_table.add_row(str(file_p.name), "[red]Failed[/red]", "-", "-")

        console.print(summary_table)
        console.print(f"\nSuccessfully investigated {successful}/{len(batch_results)} embeddings")

        if output_path:
            console.print(f"Results saved in: [cyan]{output_path.resolve()}[/cyan]")

    except Exception as e:
        console.print(f"❌ [red]Batch Investigation Error:[/red] {str(e)}")
        if verbose:
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise typer.Exit(1)


# Web UI Command
try:
    from ..web.main import main as web_main_runner

    @app.command()
    def serve(
        host: Annotated[str, typer.Option("--host", help="Host to bind")] = "127.0.0.1",
        port: Annotated[int, typer.Option("--port", help="Port to bind")] = 8000,
        database: Annotated[str, typer.Option("--database", help="Database file path")] = "perquire.db",
        reload: Annotated[bool, typer.Option("--reload", help="Enable auto-reload")] = False,
    ):
        """
        🌐 Launch the Perquire web interface.
        """
        console.print(f"🚀 Launching Perquire web interface on http://{host}:{port}")
        import sys
        original_argv = sys.argv
        sys.argv = ["perquire-web", "--host", host, "--port", str(port), "--database", database]
        if reload:
            sys.argv.append("--reload")
        try:
            web_main_runner()
        except ImportError as e:
            if "watchfiles" in str(e).lower() and reload:
                console.print("[red]Error: --reload requires 'watchfiles'. Install: uv add watchfiles[/red]")
            else:
                console.print(f"[red]Missing dependency: {e}[/red]")
                console.print("[yellow]Install web dependencies: uv add perquire[web][/yellow]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
        finally:
            sys.argv = original_argv

except ImportError:
    @app.command()
    def serve():
        """🌐 Launch the Perquire web interface (DISABLED - dependencies missing)."""
        console.print("[yellow]Web interface not available. Install: uv add perquire[web][/yellow]")


# Demo Command
try:
    from .demo import text_demo as demo_text_command

    demo_app = typer.Typer(help="Commands for demonstrating Perquire's capabilities")
    demo_app.command("text")(demo_text_command)
    app.add_typer(demo_app, name="demo")

except ImportError:
    @app.command()
    def demo():
        """Access demo commands (DISABLED - components missing)."""
        console.print("[yellow]Demo commands unavailable. Ensure all core components are installed.[/yellow]")


if __name__ == '__main__':
    app()
