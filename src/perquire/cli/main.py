"""
Main CLI interface for Perquire.
"""

import click
import rich
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm
import numpy as np
import json
import os
from pathlib import Path
from typing import Optional, List

from ..providers import list_available_providers, ProviderNotInstalledError

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="perquire")
def cli():
    """
    Perquire: Reverse Embedding Search Through Systematic Questioning
    
    A revolutionary AI system that investigates mysterious embeddings 
    through systematic questioning to uncover what they represent.
    """
    pass


# === INVESTIGATION COMMANDS DISABLED FOR LEAN INSTALLATION ===
# The following commands require the full investigation engine
# Install with: pip install perquire[api-gemini] to enable these commands:
#
# - perquire investigate <file>     # Single embedding investigation  
# - perquire batch <directory>      # Batch investigation
# - perquire status                 # Investigation history
# - perquire export                 # Export results
#
# Currently available commands:
# - perquire providers              # List available providers
# - perquire configure              # Configure settings
    """
    Investigate multiple embeddings from directory.
    
    EMBEDDINGS_DIR: Directory containing embedding files
    """
    try:
        # Find embedding files
        embedding_files = list_embedding_files(embeddings_dir, format, limit)
        
        if not embedding_files:
            console.print(f"❌ [red]No embedding files found in:[/red] {embeddings_dir}")
            return
        
        console.print(f"📁 [bold]Found {len(embedding_files)} embedding files[/bold]")
        
        if not Confirm.ask(f"Proceed with batch investigation?"):
            console.print("⏹️ [yellow]Operation cancelled[/yellow]")
            return
        
        results = []
        
        with Progress(console=console) as progress:
            task = progress.add_task("Processing embeddings...", total=len(embedding_files))
            
            for i, file_path in enumerate(embedding_files):
                try:
                    # Load embedding
                    embedding = load_embedding_from_file(file_path, format)
                    
                    # Investigate
                    if ensemble:
                        investigator = create_ensemble_investigator(
                            strategies=['default', 'artistic', 'scientific'],
                            database_path=database
                        )
                        result = investigator.investigate(
                            target_embedding=embedding,
                            parallel=parallel,
                            save_ensemble_result=True,
                            verbose=False
                        )
                    else:
                        investigator = create_investigator(
                            llm_provider=provider,
                            strategy=strategy,
                            database_path=database
                        )
                        result = investigator.investigate(
                            target_embedding=embedding,
                            save_to_database=True,
                            verbose=False
                        )
                    
                    results.append((file_path, result))
                    progress.update(task, advance=1, description=f"Processed {file_path.name}")
                    
                except Exception as e:
                    console.print(f"❌ [red]Failed to process {file_path}:[/red] {str(e)}")
                    progress.update(task, advance=1)
        
        # Display summary
        display_batch_results(results)
        
    except Exception as e:
        console.print(f"❌ [red]Batch investigation failed:[/red] {str(e)}")
        raise click.Abort()


@cli.command()
@click.option('--provider', '-p', 
              type=click.Choice(['gemini', 'openai', 'anthropic', 'ollama']),
              help='Set default LLM provider')
@click.option('--api-key', '-k',
              help='Set API key for provider')
@click.option('--database', '-d',
              help='Set default database path')
@click.option('--show', is_flag=True,
              help='Show current configuration')
def configure(provider, api_key, database, show):
    """
    Configure Perquire settings.
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


@cli.command()
def providers():
    """
    List available LLM and embedding providers and their installation status.
    """
    try:
        providers = list_available_providers()
        
        console.print("\n[bold]📋 Available Providers[/bold]\n")
        
        # Embedding providers
        console.print("[bold blue]🔍 Embedding Providers[/bold blue]")
        embed_table = Table(show_header=True, header_style="bold magenta")
        embed_table.add_column("Provider", style="cyan")
        embed_table.add_column("Status", justify="center")
        embed_table.add_column("Install Command", style="dim")
        
        for name, info in providers["embedding"].items():
            status = "✅ Installed" if info["installed"] else "❌ Not installed"
            install_cmd = f"pip install perquire[{info['extra']}]" if not info["installed"] else "-"
            embed_table.add_row(name, status, install_cmd)
        
        console.print(embed_table)
        console.print()
        
        # LLM providers  
        console.print("[bold blue]🤖 LLM Providers[/bold blue]")
        llm_table = Table(show_header=True, header_style="bold magenta")
        llm_table.add_column("Provider", style="cyan")
        llm_table.add_column("Status", justify="center")
        llm_table.add_column("Install Command", style="dim")
        
        for name, info in providers["llm"].items():
            status = "✅ Installed" if info["installed"] else "❌ Not installed"
            install_cmd = f"pip install perquire[{info['extra']}]" if not info["installed"] else "-"
            llm_table.add_row(name, status, install_cmd)
        
        console.print(llm_table)
        
        # Installation examples
        console.print("\n[bold yellow]💡 Common Installation Examples:[/bold yellow]")
        console.print("   pip install perquire[api-openai,dev]     # OpenAI + dev tools")
        console.print("   pip install perquire[api-gemini,web]     # Gemini + web interface")
        console.print("   pip install perquire[local-embeddings]   # Local inference (heavy!)")
        console.print("   pip install perquire[api-openai,api-anthropic,web,dev]  # Full setup")
        
    except Exception as e:
        console.print(f"[red]Error listing providers: {e}[/red]")


@cli.command()
@click.option('--database', '-d', default='perquire.db',
              help='Database file path')
def status(database):
    """
    Show investigation status and statistics.
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


@cli.command()
@click.option('--database', '-d', default='perquire.db',
              help='Database file path')
@click.option('--output', '-o', default='investigations.json',
              help='Output file path')
@click.option('--format', '-f', default='json',
              type=click.Choice(['json', 'csv', 'txt']),
              help='Export format')
@click.option('--limit', '-l', type=int,
              help='Limit number of records')
def export(database, output, format, limit):
    """
    Export investigation results.
    """
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

# --- investigate command and its helpers ---
from ..core.investigator import PerquireInvestigator
# Assuming ProviderNotInstalledError and PerquireException are in perquire.exceptions
# from ..exceptions import ProviderNotInstalledError, PerquireException # Already imported at top
from typing import Any # For InvestigationResult type hint

# --- Configuration Helper ---
def get_global_config() -> dict:
    """Loads global configuration from ~/.perquire/config.json."""
    config_file = Path.home() / '.perquire' / 'config.json'
    config_data = {}
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
        except json.JSONDecodeError:
            console.print(f"[yellow]Warning: Could not parse config file at {config_file}. Using defaults/CLI options.[/yellow]")
    return config_data

# --- Investigator Creation Helper ---
def create_investigator_from_cli_options(
    llm_provider_name: Optional[str],
    embedding_provider_name: Optional[str],
    strategy_name: Optional[str],
    database_path_cli: Optional[str],
    verbose: bool = False
) -> "PerquireInvestigator":
    from ..llm import provider_registry as llm_registry
    from ..embeddings import embedding_registry
    # PerquireInvestigator already imported

    global_config = get_global_config()

    final_llm_provider_name = llm_provider_name or global_config.get("default_llm_provider")
    if not final_llm_provider_name:
        available_llms = llm_registry.list_providers()
        if available_llms:
            final_llm_provider_name = available_llms[0]
            if verbose: console.print(f"[dim]LLM provider not specified, defaulting to first available: [cyan]{final_llm_provider_name}[/cyan][/dim]")
        else:
            raise click.UsageError("No LLM provider available or specified. Configure one or ensure API keys are set.")

    final_embedding_provider_name = embedding_provider_name or global_config.get("default_embedding_provider")
    if not final_embedding_provider_name:
        available_embeddings = embedding_registry.list_providers()
        if available_embeddings:
            final_embedding_provider_name = available_embeddings[0]
            if verbose: console.print(f"[dim]Embedding provider not specified, defaulting to first available: [cyan]{final_embedding_provider_name}[/cyan][/dim]")
        else:
            raise click.UsageError("No embedding provider available or specified. Configure one or ensure API keys are set.")

    db_path = database_path_cli or global_config.get('default_database')
    db_provider_instance = None
    if db_path:
        try:
            from ..database.duckdb_provider import DuckDBProvider
            from ..database.base import DatabaseConfig
            db_config = DatabaseConfig(connection_string=str(db_path))
            db_provider_instance = DuckDBProvider(db_config)
            if verbose: console.print(f"[dim]Using database: {Path(db_path).resolve()}[/dim]", style="dim")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not initialize database at {db_path}: {e}. Results may not be saved.[/yellow]")
    elif verbose:
        console.print("[dim]Database path not specified. Results will not be saved to a persistent database.[/dim]", style="dim")

    try:
        investigator = PerquireInvestigator(
            llm_provider=final_llm_provider_name,
            embedding_provider=final_embedding_provider_name,
            database_provider=db_provider_instance,
        )
        if verbose:
            console.print(f"[dim]Investigator created with LLM: [cyan]{final_llm_provider_name}[/cyan], Embeddings: [cyan]{final_embedding_provider_name}[/cyan][/dim]")
        return investigator
    except ProviderNotInstalledError as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print(f"👉 Please install the required provider and try again.")
        raise click.Abort()
    except PerquireException as e: # Catch specific Perquire exceptions
        console.print(f"[red]Error creating investigator: {e}[/red]")
        raise click.Abort()
    except Exception as e: # Catch any other unexpected error during instantiation
        console.print(f"[red]Unexpected error creating investigator: {e}[/red]")
        if verbose: import traceback; console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise click.Abort()

# --- Display Helper ---
def display_investigation_result(result: Any, verbose: bool = False):
    from ..core.result import InvestigationResult
    if not isinstance(result, InvestigationResult):
        console.print(f"[yellow]Cannot display result: Expected InvestigationResult, got {type(result)}[/yellow]")
        return

    console.print("\n[bold green]✅ Investigation Complete![/bold green]")
    console.print(f"   [bold]Description:[/bold] {result.description}")
    console.print(f"   [bold]Similarity:[/bold]  {result.final_similarity:.4f}")
    console.print(f"   [bold]Iterations:[/bold]  {result.iterations}")
    console.print(f"   [bold]Duration:[/bold]    {result.total_duration_seconds:.2f}s")
    console.print(f"   [bold]Strategy:[/bold]    {result.strategy_name}")

    if verbose and result.questions_history:
        console.print("\n[bold]Question History:[/bold]")
        history_table = Table(show_header=True, header_style="bold magenta", title=None) # Removed title for cleaner look
        history_table.add_column("Iter.", style="dim")
        history_table.add_column("Phase", style="cyan")
        history_table.add_column("Question", overflow="fold") # Allow question to wrap
        history_table.add_column("Similarity", style="magenta")

        for i, qr in enumerate(result.questions_history):
            history_table.add_row(str(i + 1), qr.phase, qr.question, f"{qr.similarity:.4f}")
        console.print(history_table)
    elif verbose: # Handle case where verbose is true but no history (e.g. error before history populated)
        console.print("\n[dim]No question history to display or verbose output for history is limited.[/dim]")


@cli.command() # This should be @click.command() if main.py doesn't define 'cli' group
@click.argument('embedding_file', type=click.Path(exists=True, dir_okay=False, resolve_path=True))
@click.option('--llm-provider', 'cli_llm_provider',help='LLM provider (e.g., openai, gemini). Overrides global config.')
@click.option('--embedding-provider', 'cli_embedding_provider', help='Embedding provider for questions. Overrides global config.')
@click.option('--strategy', 'cli_strategy', help='Questioning strategy name.') # TODO: Add choices from a registry
@click.option('--database', 'cli_database_path', help='Database path for results. Overrides global config.')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output.')
@click.option('--format', 'file_format', default='npy', type=click.Choice(['npy', 'json', 'txt'], case_sensitive=False), help='Embedding file format.')
def investigate(embedding_file: str, cli_llm_provider: Optional[str], cli_embedding_provider: Optional[str], cli_strategy: Optional[str], cli_database_path: Optional[str], verbose: bool, file_format: str):
    """
    Investigate a single embedding file to uncover what it represents.

    EMBEDDING_FILE: Path to the embedding file (e.g., .npy, .json, .txt).
    """
    if verbose: console.print(f"[bold blue]🔎 Starting investigation for: {Path(embedding_file).name}[/bold blue]", style="dim")
    else: console.print(f"🔎 Investigating {Path(embedding_file).name}...")

    try:
        investigator_instance = create_investigator_from_cli_options(
            cli_llm_provider, cli_embedding_provider, cli_strategy, cli_database_path, verbose=verbose
        )

        target_embedding = load_embedding_from_file(Path(embedding_file), file_format)

        from rich.progress import Progress, SpinnerColumn, TextColumn # Local import for Progress
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console, transient=True) as pb:
            task_id = pb.add_task("Investigating...", total=None) # Indeterminate progress

            investigation_result = investigator_instance.investigate(
                target_embedding=target_embedding,
                verbose=verbose
            )
            pb.update(task_id, completed=True, description="Investigation complete.")

        display_investigation_result(investigation_result, verbose)

    except PerquireException as e:
        console.print(f"❌ [red]Investigation Error:[/red] {e}")
        if verbose: import traceback; console.print(f"[dim]{traceback.format_exc()}[/dim]")
    except Exception as e:
        console.print(f"❌ [red]An unexpected error occurred during investigation:[/red] {str(e)}")
        if verbose: import traceback; console.print(f"[dim]{traceback.format_exc()}[/dim]")
        # raise click.Abort() # Optional: Abort for unexpected errors

# --- End of investigate command and its helpers ---

# --- Batch command and its helpers ---
from rich.progress import BarColumn # Already have SpinnerColumn, TextColumn from investigate's Progress
from rich.prompt import Confirm # Confirm might be used by batch

def display_batch_summary(results: List[tuple[Path, Any]], output_dir_obj: Optional[Path] = None):
    from ..core.result import InvestigationResult # Import here
    if not results:
        console.print("[yellow]No results from batch investigation to display.[/yellow]")
        return

    console.print(f"\n[bold green]📊 Batch Investigation Summary ({len(results)} files processed)[/bold green]")

    summary_table = Table(title="Batch Results", show_header=True, header_style="bold magenta")
    summary_table.add_column("File", style="cyan", overflow="fold", max_width=50)
    summary_table.add_column("Description", overflow="fold")
    summary_table.add_column("Similarity", style="magenta", justify="right")
    summary_table.add_column("Iterations", style="blue", justify="right")

    successful_investigations = 0
    for file_path, result_item in results:
        if isinstance(result_item, InvestigationResult):
            summary_table.add_row(
                str(file_path.name),
                result_item.description,
                f"{result_item.final_similarity:.4f}",
                str(result_item.iterations)
            )
            successful_investigations += 1
        else:
            summary_table.add_row(str(file_path.name), "[red]Failed or No Result[/red]", "-", "-")

    console.print(summary_table)
    console.print(f"\nSuccessfully investigated {successful_investigations}/{len(results)} embeddings.")

    if output_dir_obj:
        console.print(f"Individual JSON results saved in: [cyan]{output_dir_obj.resolve()}[/cyan]")


@cli.command("batch") # Attached to main.py's cli group for now
@click.argument('embeddings_dir', type=click.Path(exists=True, file_okay=False, resolve_path=True))
@click.option('--llm-provider', 'cli_llm_provider', help='LLM provider. Overrides global config.')
@click.option('--embedding-provider', 'cli_embedding_provider', help='Embedding provider for questions. Overrides global config.')
@click.option('--strategy', 'cli_strategy', help='Questioning strategy name.')
@click.option('--database', 'cli_database_path', help='Database path. Overrides global config.')
@click.option('--limit', '-l', type=int, help='Limit number of files to process.')
@click.option('--format', 'file_format', default='npy', type=click.Choice(['npy', 'json', 'txt'], case_sensitive=False), help='Embedding file format.')
@click.option('--output-dir', 'output_dir_str', help="Directory to save individual JSON results for each embedding.")
@click.option('--verbose', '-v', is_flag=True, help='Enable detailed output for each investigation during the batch.')
def batch(
    embeddings_dir: str,
    cli_llm_provider: Optional[str],
    cli_embedding_provider: Optional[str],
    cli_strategy: Optional[str],
    cli_database_path: Optional[str],
    limit: Optional[int],
    file_format: str,
    output_dir_str: Optional[str],
    verbose: bool
):
    """
    Investigate multiple embedding files from a specified directory.

    EMBEDDINGS_DIR: Directory containing embedding files (e.g., .npy, .json, .txt).
    """
    dir_path = Path(embeddings_dir)
    console.print(f"[bold blue]🚀 Starting batch investigation in: {dir_path}[/bold blue]")

    try:
        investigator_instance = create_investigator_from_cli_options(
            cli_llm_provider, cli_embedding_provider, cli_strategy, cli_database_path, verbose=False # Overall batch op verbose is separate
        )

        embedding_files = list_embedding_files(dir_path, file_format, limit)
        if not embedding_files:
            console.print(f"[yellow]No '*{file_format}' files found in {dir_path}. Nothing to process.[/yellow]")
            return

        console.print(f"Found {len(embedding_files)} '{file_format}' files. {'Limiting to first '+str(limit) if limit else ''}")
        if not Confirm.ask("Proceed with batch investigation?", default=True):
            console.print("[yellow]Batch operation cancelled by user.[/yellow]")
            return

        batch_results_list = []
        output_path_obj = Path(output_dir_str) if output_dir_str else None
        if output_path_obj:
            output_path_obj.mkdir(parents=True, exist_ok=True)

        from rich.progress import Progress, SpinnerColumn, TextColumn # Re-import if not at top level of file for Progress

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}% ({task.completed}/{task.total})"),
            console=console
        ) as progress_bar:
            batch_task_id = progress_bar.add_task("Batch processing...", total=len(embedding_files))

            for file_p in embedding_files:
                progress_bar.update(batch_task_id, description=f"Processing: {file_p.name}")
                try:
                    emb = load_embedding_from_file(file_p, file_format)
                    res = investigator_instance.investigate(
                        target_embedding=emb,
                        verbose=verbose # Pass the main verbose flag for individual investigations
                    )
                    batch_results_list.append((file_p, res))

                    if output_path_obj and res and hasattr(res, 'to_dict'):
                        try:
                            with open(output_path_obj / f"{file_p.stem}_result.json", 'w') as f_out:
                                json.dump(res.to_dict(), f_out, indent=2, default=str)
                        except Exception as e_json:
                            console.print(f"[red]Error saving JSON for {file_p.name}: {e_json}[/red]")

                    if verbose and res: # If main verbose is on, show individual full results
                        console.rule(f"Result for {file_p.name}")
                        display_investigation_result(res, verbose=True)
                        console.rule()

                except Exception as e_item:
                    console.print(f"[red]Failed processing {file_p.name}: {e_item}[/red]")
                    batch_results_list.append((file_p, None))
                finally:
                    progress_bar.update(batch_task_id, advance=1)

        display_batch_summary(batch_results_list, output_path_obj)

    except PerquireException as e:
        console.print(f"❌ [red]Batch Investigation Error:[/red] {e}")
        if verbose: import traceback; console.print(f"[dim]{traceback.format_exc()}[/dim]")
    except Exception as e:
        console.print(f"❌ [red]An unexpected error occurred during batch investigation:[/red] {str(e)}")
        if verbose: import traceback; console.print(f"[dim]{traceback.format_exc()}[/dim]")

# --- End of Batch command ---

def load_embedding_from_file(file_path: Path, format: str) -> np.ndarray:
    """Load embedding from file based on format."""
    if format == 'npy':
        return np.load(file_path)
    elif format == 'json':
        with open(file_path, 'r') as f:
            data = json.load(f)
            return np.array(data)
    elif format == 'txt':
        return np.loadtxt(file_path)
    else:
        raise ValueError(f"Unsupported format: {format}")


def list_embedding_files(directory: Path, format: str, limit: Optional[int] = None) -> List[Path]:
    """List embedding files in directory."""
    directory = Path(directory)
    
    if format == 'npy':
        pattern = '*.npy'
    elif format == 'json':
        pattern = '*.json'
    elif format == 'txt':
        pattern = '*.txt'
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    files = list(directory.glob(pattern))
    
    if limit:
        files = files[:limit]
    
    return files


# === HEAVY FUNCTIONS DISABLED FOR LEAN INSTALLATION ===
# These functions require the full investigation engine and are disabled
# Install with: pip install perquire[api-gemini] to enable full functionality

# def display_investigation_result(result):
# def display_batch_results(results: List[tuple]):
# ... (functions commented out - require heavy dependencies)


if __name__ == '__main__':
    cli()