#!/usr/bin/env python3
"""
Perquire CLI Upgrade Demonstration
Shows the before/after of Click → Typer migration
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.columns import Columns
from rich.syntax import Syntax
import time

console = Console()

def show_intro():
    """Show introduction."""
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]Perquire CLI Upgrade Demonstration[/bold cyan]\n\n"
        "[yellow]Migration:[/yellow] Click → Typer\n"
        "[green]Python:[/green] 3.8+ → 3.12+\n"
        "[blue]Benefits:[/blue] Type safety, Rich integration, Modern Python",
        title="🚀 Upgrade Complete",
        border_style="cyan"
    ))
    time.sleep(1)

def show_before_after():
    """Show code comparison."""
    console.print("\n[bold blue]📝 Code Comparison[/bold blue]\n")

    before = '''# Before (Click)
@click.command()
@click.argument('file', type=click.Path(exists=True))
@click.option('--verbose', '-v', is_flag=True)
def investigate(file, verbose):
    """Investigate an embedding."""
    console.print(f"Investigating {file}...")
    # ...
'''

    after = '''# After (Typer)
@app.command()
def investigate(
    embedding_file: Annotated[
        Path,
        typer.Argument(
            help="Path to embedding file",
            exists=True
        )
    ],
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose", "-v",
            help="Enable verbose output"
        )
    ] = False,
):
    """🔎 Investigate an embedding."""
    console.print(f"🔎 Investigating {embedding_file.name}...")
    # ...
'''

    before_syntax = Syntax(before, "python", theme="monokai", line_numbers=True)
    after_syntax = Syntax(after, "python", theme="monokai", line_numbers=True)

    columns = Columns([
        Panel(before_syntax, title="[red]Before (Click)[/red]", border_style="red"),
        Panel(after_syntax, title="[green]After (Typer)[/green]", border_style="green")
    ], equal=True, expand=True)

    console.print(columns)
    time.sleep(2)

def show_benefits():
    """Show benefits table."""
    console.print("\n[bold blue]✨ Key Benefits[/bold blue]\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Feature", style="cyan", width=25)
    table.add_column("Before (Click)", style="red", width=30)
    table.add_column("After (Typer)", style="green", width=30)

    table.add_row(
        "Type Safety",
        "❌ String-based",
        "✅ Type hints with Annotated"
    )
    table.add_row(
        "IDE Support",
        "⚠️  Limited autocomplete",
        "✅ Full autocomplete & validation"
    )
    table.add_row(
        "Rich Integration",
        "⚠️  Manual setup",
        "✅ Native, automatic"
    )
    table.add_row(
        "Validation",
        "⚠️  Runtime only",
        "✅ Pydantic + runtime"
    )
    table.add_row(
        "Modern Python",
        "❌ 3.7+ patterns",
        "✅ 3.12+ patterns"
    )
    table.add_row(
        "Error Messages",
        "⚠️  Basic",
        "✅ Rich, colorful, helpful"
    )

    console.print(table)
    time.sleep(2)

def show_commands():
    """Show available commands."""
    console.print("\n[bold blue]🎯 Available Commands[/bold blue]\n")

    commands_table = Table(show_header=True, header_style="bold magenta")
    commands_table.add_column("#", style="dim", width=3)
    commands_table.add_column("Command", style="cyan", width=15)
    commands_table.add_column("Description", style="white")

    commands = [
        ("providers", "📋 List available LLM and embedding providers"),
        ("configure", "⚙️  Configure Perquire settings"),
        ("status", "📊 Show investigation status and statistics"),
        ("export", "📤 Export investigation results"),
        ("investigate", "🔎 Investigate a single embedding file"),
        ("batch", "🚀 Investigate multiple embedding files"),
        ("serve", "🌐 Launch the Perquire web interface"),
        ("demo", "🎮 Access demo commands"),
    ]

    for i, (cmd, desc) in enumerate(commands, 1):
        commands_table.add_row(str(i), cmd, desc)

    console.print(commands_table)
    time.sleep(2)

def show_structure():
    """Show new structure."""
    console.print("\n[bold blue]🏗️  New Structure[/bold blue]\n")

    structure = """
perquire/
├── pyproject.toml          ← Python 3.12+, Typer, package=true
├── src/perquire/cli/
│   └── main.py            ← 616 lines of type-safe Typer code
└── UPGRADE_SUMMARY.md     ← Complete documentation

Key Changes:
• requires-python = ">=3.12"  (was >=3.8)
• typer>=0.12.0               (was click>=8.2.1)
• [tool.uv] package = true    (new)
• [build-system] hatchling    (new)
• entry_point: ...main:app    (was :cli)
"""

    console.print(Panel(structure, title="📁 Project Structure", border_style="blue"))
    time.sleep(2)

def show_usage_examples():
    """Show usage examples."""
    console.print("\n[bold blue]💡 Usage Examples[/bold blue]\n")

    examples = [
        ("perquire --help", "Show all commands"),
        ("perquire --version", "Show version (0.2.0)"),
        ("perquire providers", "List LLM/embedding providers"),
        ("perquire configure --show", "View configuration"),
        ("perquire investigate embedding.npy", "Investigate single file"),
        ("perquire batch embeddings/ --limit 10", "Batch investigate"),
        ("perquire serve --port 8080", "Start web server"),
    ]

    for cmd, desc in examples:
        console.print(f"  [cyan]${cmd}[/cyan]")
        console.print(f"    [dim]{desc}[/dim]\n")

    time.sleep(2)

def show_installation():
    """Show installation."""
    console.print("\n[bold blue]📦 Installation[/bold blue]\n")

    install_code = """# Clone repository
git clone https://github.com/franklinbaldo/perquire
cd perquire

# Install with uv (recommended)
uv sync

# Run commands
uv run perquire --help
uv run perquire providers

# Or install with pip
pip install -e .
perquire --help
"""

    console.print(Panel(
        Syntax(install_code, "bash", theme="monokai"),
        title="Installation",
        border_style="green"
    ))
    time.sleep(2)

def show_stats():
    """Show change statistics."""
    console.print("\n[bold blue]📈 Change Statistics[/bold blue]\n")

    stats_table = Table(show_header=True, header_style="bold magenta")
    stats_table.add_column("File", style="cyan")
    stats_table.add_column("Changes", justify="right", style="yellow")
    stats_table.add_column("Impact", style="green")

    stats_table.add_row("pyproject.toml", "+19 lines", "Config & dependencies")
    stats_table.add_row("main.py", "+698 / -440", "Complete Typer rewrite")
    stats_table.add_row("UPGRADE_SUMMARY.md", "+315 lines", "Documentation")
    stats_table.add_row("[bold]Total[/bold]", "[bold]+1032 / -440[/bold]", "[bold]Major upgrade[/bold]")

    console.print(stats_table)
    time.sleep(2)

def main():
    """Run demonstration."""
    console.clear()

    show_intro()
    show_before_after()
    show_benefits()
    show_commands()
    show_structure()
    show_usage_examples()
    show_installation()
    show_stats()

    console.print("\n" + "="*70)
    console.print("[bold green]✅ Perquire CLI Upgrade Complete![/bold green]".center(70))
    console.print("="*70 + "\n")

    console.print("[bold cyan]Summary:[/bold cyan]")
    console.print("  • Migrated from Click to Typer")
    console.print("  • Upgraded to Python 3.12+")
    console.print("  • Added full type safety")
    console.print("  • Enhanced with Rich UI")
    console.print("  • Maintained backward compatibility")
    console.print("\n[bold yellow]Repository:[/bold yellow] https://github.com/franklinbaldo/perquire")
    console.print("[bold yellow]Commit:[/bold yellow] 32292e3\n")

if __name__ == "__main__":
    main()
