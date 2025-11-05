#!/usr/bin/env python3
"""
Quick test to verify CLI structure without dependencies
"""
import sys
import ast
from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()

def analyze_cli():
    """Analyze the CLI file structure."""
    cli_file = Path("src/perquire/cli/main.py")

    if not cli_file.exists():
        console.print("[red]CLI file not found![/red]")
        return False

    content = cli_file.read_text()

    # Parse the AST
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        console.print(f"[red]Syntax error in CLI file: {e}[/red]")
        return False

    # Find imports
    imports = []
    typer_imported = False
    rich_imported = False

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
                if alias.name == 'typer':
                    typer_imported = True
                if 'rich' in alias.name:
                    rich_imported = True
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                if node.module == 'typer':
                    typer_imported = True
                if 'rich' in node.module:
                    rich_imported = True

    # Find function definitions (commands)
    commands = []
    app_created = False

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == 'app':
                    app_created = True

        if isinstance(node, ast.FunctionDef):
            # Check if it's decorated with @app.command()
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Attribute):
                    if isinstance(decorator.value, ast.Name) and decorator.value.id == 'app':
                        if decorator.attr == 'command':
                            commands.append(node.name)
                elif isinstance(decorator, ast.Call):
                    if isinstance(decorator.func, ast.Attribute):
                        if isinstance(decorator.func.value, ast.Name) and decorator.func.value.id == 'app':
                            if decorator.func.attr == 'command':
                                commands.append(node.name)

    # Display results
    console.print("\n[bold cyan]🔍 Perquire CLI Analysis[/bold cyan]\n")

    # Framework check
    framework_table = Table(title="Framework")
    framework_table.add_column("Check", style="cyan")
    framework_table.add_column("Status", style="green")

    framework_table.add_row("Typer imported", "✅ Yes" if typer_imported else "❌ No")
    framework_table.add_row("Rich imported", "✅ Yes" if rich_imported else "❌ No")
    framework_table.add_row("Typer app created", "✅ Yes" if app_created else "❌ No")

    console.print(framework_table)
    console.print()

    # Commands check
    if commands:
        commands_table = Table(title="Commands Found")
        commands_table.add_column("#", style="dim")
        commands_table.add_column("Command", style="cyan")

        for i, cmd in enumerate(commands, 1):
            commands_table.add_row(str(i), cmd)

        console.print(commands_table)
        console.print(f"\n[green]✅ Found {len(commands)} commands[/green]")
    else:
        console.print("[yellow]⚠️  No commands found[/yellow]")

    # Type annotations check
    has_annotations = 'Annotated' in content
    console.print(f"\n[bold]Type Annotations:[/bold] {'✅ Yes' if has_annotations else '❌ No'}")

    # Summary
    console.print("\n[bold green]✅ CLI Structure Analysis Complete![/bold green]")
    console.print(f"   • Framework: [cyan]Typer[/cyan] with [cyan]Rich[/cyan]")
    console.print(f"   • Commands: [cyan]{len(commands)}[/cyan]")
    console.print(f"   • Type-safe: [cyan]{'Yes' if has_annotations else 'No'}[/cyan]")

    return typer_imported and app_created and len(commands) > 0

if __name__ == "__main__":
    success = analyze_cli()
    sys.exit(0 if success else 1)
