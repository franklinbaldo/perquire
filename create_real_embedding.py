#!/usr/bin/env python3
"""
Create a REAL embedding using Gemini API
"""
import os
import sys
import json
import numpy as np
from pathlib import Path

# Add to path
sys.path.insert(0, 'src')

from rich.console import Console
from rich.panel import Panel

console = Console()

def create_real_embedding():
    """Create real embedding with Gemini."""
    try:
        import google.generativeai as genai

        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            console.print("[red]❌ No API key found[/red]")
            return False

        genai.configure(api_key=api_key)

        # Test text
        text = "The bittersweet feeling of nostalgia when looking at old photographs"

        console.print("\n" + "="*70)
        console.print(Panel.fit(
            "[bold cyan]Creating Real Embedding with Gemini API[/bold cyan]\n\n"
            f"[yellow]Text:[/yellow] \"{text}\"",
            title="🚀 Real Demo",
            border_style="cyan"
        ))
        console.print("="*70 + "\n")

        console.print("[cyan]📡 Calling Gemini API...[/cyan]")

        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_document"
        )

        embedding = np.array(result['embedding'])

        # Save as numpy
        output_dir = Path("test_embeddings")
        output_dir.mkdir(exist_ok=True)

        npy_file = output_dir / "nostalgia_real.npy"
        np.save(npy_file, embedding)

        # Also save as JSON for inspection
        json_file = output_dir / "nostalgia_real.json"
        with open(json_file, 'w') as f:
            json.dump(embedding.tolist(), f)

        console.print(f"\n[green]✅ Real embedding created![/green]")
        console.print(f"  • Text: \"{text}\"")
        console.print(f"  • Shape: {embedding.shape}")
        console.print(f"  • Dimensions: {len(embedding)}")
        console.print(f"  • Model: text-embedding-004")
        console.print(f"  • NumPy file: {npy_file}")
        console.print(f"  • JSON file: {json_file}")
        console.print(f"  • Size: {embedding.nbytes} bytes\n")

        # Show preview
        console.print("[bold blue]Preview (first 10 values):[/bold blue]")
        console.print(f"  {embedding[:10].tolist()}\n")

        console.print("[bold green]🎯 Now you can investigate it:[/bold green]")
        console.print(f"  [cyan]source /home/frank/workspace/.envrc[/cyan]")
        console.print(f"  [cyan]uv run perquire investigate {npy_file} --verbose[/cyan]\n")

        return True

    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        return False

if __name__ == "__main__":
    success = create_real_embedding()
    sys.exit(0 if success else 1)
