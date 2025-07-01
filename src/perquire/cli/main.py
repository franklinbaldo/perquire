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