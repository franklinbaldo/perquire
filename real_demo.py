#!/usr/bin/env python3
"""
Real Perquire Demo with Actual API Call
"""
import os
import sys
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
import google.generativeai as genai

console = Console()

def create_test_embedding(text: str, output_file: Path):
    """Create a real embedding using Gemini API."""
    try:
        # Configure API
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            console.print("[red]❌ No API key found in environment[/red]")
            return False

        genai.configure(api_key=api_key)

        console.print(f"\n[cyan]Creating embedding for:[/cyan]")
        console.print(f"  [italic]\"{text}\"[/italic]\n")

        # Create embedding
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_document"
        )

        embedding = np.array(result['embedding'])

        # Save to file
        np.save(output_file, embedding)

        console.print(f"[green]✅ Embedding created:[/green]")
        console.print(f"  • Shape: {embedding.shape}")
        console.print(f"  • File: {output_file}")
        console.print(f"  • Size: {embedding.nbytes} bytes\n")

        return True

    except Exception as e:
        console.print(f"[red]❌ Error creating embedding: {e}[/red]")
        return False


def main():
    """Run real demo."""
    console.print("\n" + "="*70)
    console.print(Panel.fit(
        "[bold cyan]Perquire Real Demo[/bold cyan]\n\n"
        "[yellow]Using actual Gemini API to create and investigate embeddings[/yellow]",
        title="🚀 Live Test",
        border_style="cyan"
    ))
    console.print("="*70 + "\n")

    # Test texts
    test_cases = [
        ("The bittersweet feeling of nostalgia when looking at old photographs", "nostalgia.npy"),
        ("A cozy coffee shop on a rainy evening with warm yellow lights", "coffee_shop.npy"),
        ("The excitement and nervousness before giving a big presentation", "presentation.npy"),
    ]

    embeddings_dir = Path("test_embeddings")
    embeddings_dir.mkdir(exist_ok=True)

    console.print(f"[bold blue]📦 Creating Test Embeddings[/bold blue]\n")

    created_files = []
    for text, filename in test_cases:
        output_file = embeddings_dir / filename
        if create_test_embedding(text, output_file):
            created_files.append((text, output_file))

    if not created_files:
        console.print("[red]❌ Failed to create any embeddings[/red]")
        return 1

    console.print(f"\n[green]✅ Created {len(created_files)} embeddings[/green]")
    console.print("\n[bold yellow]Next Steps:[/bold yellow]")
    console.print("  Run Perquire investigation with:")
    console.print(f"  [cyan]cd perquire[/cyan]")
    console.print(f"  [cyan]source /home/frank/workspace/.envrc[/cyan]")

    for text, filepath in created_files:
        console.print(f"  [cyan]uv run perquire investigate {filepath} --verbose[/cyan]")

    console.print("\n[dim]Note: Full investigation requires 'uv sync' to complete first[/dim]\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
