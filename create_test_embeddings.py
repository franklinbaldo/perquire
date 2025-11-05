#!/usr/bin/env python3
"""
Create test embeddings using simple lists (no numpy required)
"""
import json
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

console = Console()

def create_mock_embedding(text: str, output_file: Path, dim: int = 384):
    """Create a mock embedding (list of floats)."""
    # Create a deterministic "embedding" based on text
    # In reality, this would come from an embedding model
    import hashlib

    # Use text hash to create deterministic values
    text_hash = hashlib.md5(text.encode()).hexdigest()

    # Generate pseudo-random values based on hash
    embedding = []
    for i in range(dim):
        # Use hash bytes to generate values between -1 and 1
        idx = (i * 2) % len(text_hash)
        hex_pair = text_hash[idx:idx+2]
        if len(hex_pair) < 2:
            hex_pair = text_hash[0:2]
        byte_val = int(hex_pair, 16)
        normalized = (byte_val / 255.0) * 2 - 1  # Scale to [-1, 1]
        embedding.append(round(normalized, 6))

    # Save as JSON
    with open(output_file, 'w') as f:
        json.dump(embedding, f)

    return embedding


def main():
    """Create test embeddings."""
    console.print("\n" + "="*70)
    console.print(Panel.fit(
        "[bold cyan]Creating Test Embeddings[/bold cyan]\n\n"
        "[yellow]Generating mock embeddings for demonstration[/yellow]",
        title="🎯 Test Data",
        border_style="cyan"
    ))
    console.print("="*70 + "\n")

    # Test texts
    test_cases = [
        ("The bittersweet feeling of nostalgia when looking at old photographs", "nostalgia.json"),
        ("A cozy coffee shop on a rainy evening with warm yellow lights", "coffee_shop.json"),
        ("The excitement and nervousness before giving a big presentation", "presentation.json"),
        ("Ancient redwood trees towering in misty morning fog", "redwood_forest.json"),
        ("The satisfaction of completing a difficult coding challenge", "coding_satisfaction.json"),
    ]

    embeddings_dir = Path("test_embeddings")
    embeddings_dir.mkdir(exist_ok=True)

    console.print(f"[bold blue]📦 Creating Mock Embeddings[/bold blue]\n")

    created = []
    for text, filename in test_cases:
        output_file = embeddings_dir / filename

        console.print(f"[cyan]Creating:[/cyan] {filename}")
        console.print(f"  [dim]Text: \"{text[:50]}...\"[/dim]")

        embedding = create_mock_embedding(text, output_file)

        console.print(f"  [green]✅ Saved:[/green] {output_file}")
        console.print(f"  [dim]Shape: ({len(embedding)},)[/dim]\n")

        created.append((text, output_file))

    console.print(f"\n[green]✅ Created {len(created)} test embeddings[/green]")
    console.print(f"[dim]Location: {embeddings_dir.absolute()}[/dim]\n")

    # Show how to use them
    console.print("[bold yellow]🚀 Next: Test Perquire CLI[/bold yellow]\n")

    console.print("[cyan]Option 1: Test single investigation[/cyan]")
    console.print(f"  cd perquire")
    console.print(f"  source /home/frank/workspace/.envrc")
    console.print(f"  uv run perquire investigate test_embeddings/nostalgia.json --format json --verbose\n")

    console.print("[cyan]Option 2: Test batch investigation[/cyan]")
    console.print(f"  cd perquire")
    console.print(f"  source /home/frank/workspace/.envrc")
    console.print(f"  uv run perquire batch test_embeddings/ --format json --limit 3 --verbose\n")

    console.print("[cyan]Option 3: Check providers (no API needed)[/cyan]")
    console.print(f"  cd perquire")
    console.print(f"  uv run perquire providers\n")

    console.print("[dim]Note: Full investigation with LLM requires 'uv sync' to complete[/dim]")
    console.print("[dim]      Mock embeddings are for testing CLI structure only[/dim]\n")

    # Show file contents
    console.print("[bold blue]📄 Example embedding file:[/bold blue]")
    console.print(f"[dim]$ cat {created[0][1]}[/dim]")
    with open(created[0][1], 'r') as f:
        content = f.read()
        preview = content[:200] + "..." if len(content) > 200 else content
        console.print(f"[yellow]{preview}[/yellow]\n")


if __name__ == "__main__":
    main()
