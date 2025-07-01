"""
Individual CLI commands for Perquire.
"""

import click
from rich.console import Console
from .main import investigate, batch, configure, status, export

console = Console()

# Re-export commands for modular access
__all__ = ['investigate', 'batch', 'configure', 'status', 'export']