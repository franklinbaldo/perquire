"""
Lean CLI interface for Perquire - only lightweight commands.
"""

import click
from rich.console import Console
from rich.table import Table
import json
import os
from pathlib import Path

from ..providers import list_available_providers, ProviderNotInstalledError

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="perquire")
def cli():
    """
    Perquire: Reverse Embedding Search Through Systematic Questioning
    
    Lightweight installation - install providers to enable full functionality:
    
        pip install perquire[api-gemini]    # Gemini provider
        pip install perquire[api-openai]    # OpenAI provider
        pip install perquire[web]           # Web interface
    """
    pass

# Import commands from main.py (or a commands.py if that's the final structure)
# For now, assuming investigate is in main.py and is a click.Command instance
try:
    from .main import investigate as investigate_command
    from .main import status as status_command
    from .main import batch as batch_command # Added batch command
    from .main import export as export_command # Assuming export should also be here

    cli.add_command(investigate_command, "investigate")
    cli.add_command(status_command, "status")
    cli.add_command(batch_command, "batch") # Added batch command
    cli.add_command(export_command, "export") # Added export command
except ImportError as e:
    # This allows lean_main to function even if main.py has issues or missing commands,
    # though the commands themselves wouldn't be available.
    # A better approach might be to ensure main.py is robust.
    pass # Or print a warning: print(f"Warning: Could not import all commands from main.py: {e}")


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
        
        # Available commands
        console.print("\n[bold green]🚀 Available Commands After Installation:[/bold green]")
        console.print("   perquire investigate <file>       # Investigate single embedding")
        console.print("   perquire batch <directory>        # Batch investigation")
        console.print("   perquire-web                      # Start web interface")
        
    except Exception as e:
        console.print(f"[red]Error listing providers: {e}[/red]")


@cli.command()
@click.option('--provider', '-p', help='Set default provider')
@click.option('--api-key', '-k', help='Set API key for current provider')
@click.option('--database', '-d', help='Set default database path')
@click.option('--show', '-s', is_flag=True, help='Show current configuration')
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
def info():
    """
    Show Perquire installation information.
    """
    console.print("\n[bold]ℹ️ Perquire Installation Info[/bold]\n")
    
    # Basic info
    info_table = Table()
    info_table.add_column("Property")
    info_table.add_column("Value")
    
    info_table.add_row("Version", "0.1.0")
    info_table.add_row("Installation Type", "Lean (core only)")
    info_table.add_row("Configuration", str(Path.home() / '.perquire' / 'config.json'))
    
    console.print(info_table)
    
    # Check providers
    try:
        providers = list_available_providers()
        installed_count = sum(
            1 for provider_type in providers.values() 
            for provider_info in provider_type.values() 
            if provider_info['installed']
        )
        total_count = sum(len(provider_type) for provider_type in providers.values())
        
        console.print(f"\n📦 Providers: {installed_count}/{total_count} installed")
        
        if installed_count == 0:
            console.print("\n[yellow]⚠️ No providers installed. Install at least one to enable investigations:[/yellow]")
            console.print("   pip install perquire[api-gemini]")
            
    except Exception as e:
        console.print(f"\n[red]Error checking providers: {e}[/red]")


if __name__ == '__main__':
    cli()