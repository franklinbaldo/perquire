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

from .. import create_investigator, create_ensemble_investigator, investigate_embedding
from ..core import InvestigationResult
from ..exceptions import InvestigationError

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


@cli.command()
@click.argument('embedding_file', type=click.Path(exists=True))
@click.option('--provider', '-p', default='gemini', 
              type=click.Choice(['gemini', 'openai', 'anthropic', 'ollama']),
              help='LLM provider to use')
@click.option('--strategy', '-s', default='default',
              type=click.Choice(['default', 'artistic', 'scientific', 'emotional']),
              help='Investigation strategy to use')
@click.option('--database', '-d', default='perquire.db',
              help='Database file path')
@click.option('--save/--no-save', default=True,
              help='Save results to database')
@click.option('--verbose', '-v', is_flag=True,
              help='Verbose output')
@click.option('--format', '-f', default='json',
              type=click.Choice(['json', 'npy', 'txt']),
              help='Input file format')
def investigate(embedding_file, provider, strategy, database, save, verbose, format):
    """
    Investigate a single embedding from file.
    
    EMBEDDING_FILE: Path to file containing the embedding
    """
    try:
        # Load embedding
        embedding = load_embedding_from_file(embedding_file, format)
        
        if verbose:
            console.print(f"🔍 [bold]Investigating embedding from:[/bold] {embedding_file}")
            console.print(f"📊 [bold]Embedding shape:[/bold] {embedding.shape}")
            console.print(f"🤖 [bold]Provider:[/bold] {provider}")
            console.print(f"🧠 [bold]Strategy:[/bold] {strategy}")
        
        # Create investigator
        investigator = create_investigator(
            llm_provider=provider,
            embedding_provider=provider if provider == 'gemini' else 'gemini',
            strategy=strategy,
            database_path=database
        )
        
        # Run investigation with progress
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Investigating embedding...", total=None)
            
            result = investigator.investigate(
                target_embedding=embedding,
                save_to_database=save,
                verbose=verbose
            )
        
        # Display results
        display_investigation_result(result)
        
        if save:
            console.print(f"💾 [green]Results saved to database:[/green] {database}")
        
    except Exception as e:
        console.print(f"❌ [red]Investigation failed:[/red] {str(e)}")
        raise click.Abort()


@cli.command()
@click.argument('embeddings_dir', type=click.Path(exists=True, file_okay=False))
@click.option('--provider', '-p', default='gemini',
              type=click.Choice(['gemini', 'openai', 'anthropic', 'ollama']),
              help='LLM provider to use')
@click.option('--strategy', '-s', default='default',
              type=click.Choice(['default', 'artistic', 'scientific', 'emotional']),
              help='Investigation strategy to use')
@click.option('--database', '-d', default='perquire.db',
              help='Database file path')
@click.option('--ensemble', is_flag=True,
              help='Use ensemble investigation')
@click.option('--parallel', is_flag=True, default=True,
              help='Process in parallel')
@click.option('--limit', '-l', type=int,
              help='Limit number of files to process')
@click.option('--format', '-f', default='json',
              type=click.Choice(['json', 'npy', 'txt']),
              help='Input file format')
def batch(embeddings_dir, provider, strategy, database, ensemble, parallel, limit, format):
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


def display_investigation_result(result: InvestigationResult):
    """Display investigation result in a nice format."""
    console.print()
    console.print("[bold]🔍 Investigation Results[/bold]")
    console.print()
    
    # Main info
    info_table = Table()
    info_table.add_column("Field")
    info_table.add_column("Value")
    
    info_table.add_row("Investigation ID", result.investigation_id)
    info_table.add_row("Description", result.description)
    info_table.add_row("Final Similarity", f"{result.final_similarity:.4f}")
    info_table.add_row("Iterations", str(result.iterations))
    info_table.add_row("Strategy", result.strategy_name)
    info_table.add_row("Duration", str(result.end_time - result.start_time if result.end_time else "N/A"))
    
    console.print(info_table)
    
    # Questions (if any)
    if hasattr(result, 'questions') and result.questions:
        console.print()
        questions_table = Table(title="Investigation Questions")
        questions_table.add_column("Q#")
        questions_table.add_column("Question")
        questions_table.add_column("Answer")
        questions_table.add_column("Similarity")
        
        for i, q in enumerate(result.questions[-5:], 1):  # Show last 5 questions
            questions_table.add_row(
                str(i),
                q.question[:40] + "..." if len(q.question) > 40 else q.question,
                q.answer[:40] + "..." if len(q.answer) > 40 else q.answer,
                f"{q.similarity:.3f}"
            )
        
        console.print(questions_table)


def display_batch_results(results: List[tuple]):
    """Display batch investigation results summary."""
    console.print()
    console.print("[bold]📊 Batch Investigation Summary[/bold]")
    console.print()
    
    if not results:
        console.print("❌ [red]No successful investigations[/red]")
        return
    
    # Summary stats
    similarities = [result.final_similarity for _, result in results]
    iterations = [result.iterations for _, result in results]
    
    summary_table = Table(title="Summary Statistics")
    summary_table.add_column("Metric")
    summary_table.add_column("Value")
    
    summary_table.add_row("Total Processed", str(len(results)))
    summary_table.add_row("Average Similarity", f"{np.mean(similarities):.3f}")
    summary_table.add_row("Max Similarity", f"{max(similarities):.3f}")
    summary_table.add_row("Min Similarity", f"{min(similarities):.3f}")
    summary_table.add_row("Average Iterations", f"{np.mean(iterations):.1f}")
    
    console.print(summary_table)
    
    # Top results
    sorted_results = sorted(results, key=lambda x: x[1].final_similarity, reverse=True)
    
    console.print()
    top_table = Table(title="Top 5 Results")
    top_table.add_column("File")
    top_table.add_column("Description")
    top_table.add_column("Similarity")
    
    for file_path, result in sorted_results[:5]:
        top_table.add_row(
            file_path.name,
            result.description[:50] + "..." if len(result.description) > 50 else result.description,
            f"{result.final_similarity:.3f}"
        )
    
    console.print(top_table)


if __name__ == '__main__':
    cli()