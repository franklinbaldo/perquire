#!/usr/bin/env python3
"""
Quick Perquire Demo
Demonstrates how Perquire investigates unknown embeddings through systematic questioning
"""

import os
import sys

# Add perquire to path
sys.path.insert(0, '/home/frank/workspace/perquire/src')

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

def demo_conceptual_explanation():
    """Explain what Perquire does conceptually."""

    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]Perquire: Reverse Embedding Investigation[/bold cyan]\n\n"
        "[yellow]Traditional Search:[/yellow] \"Find embeddings matching my query\"\n"
        "[green]Perquire:[/green] \"What query would create this embedding?\"\n\n"
        "[dim]Like a detective investigating a mysterious footprint,[/dim]\n"
        "[dim]Perquire asks strategic questions until it uncovers the story.[/dim]",
        title="🔍 What is Perquire?"
    ))

def demo_investigation_process():
    """Show how the investigation process works."""

    console.print("\n[bold blue]📊 Investigation Process Example[/bold blue]\n")

    # Simulated investigation steps
    steps = [
        {
            "phase": "🌍 Exploration",
            "question": "Does this relate to human emotions or physical objects?",
            "similarity": 0.45,
            "status": "Getting warmer..."
        },
        {
            "phase": "🌍 Exploration",
            "question": "Is this about positive or negative feelings?",
            "similarity": 0.62,
            "status": "Progress!"
        },
        {
            "phase": "🎯 Refinement",
            "question": "Does this involve memory or anticipation?",
            "similarity": 0.78,
            "status": "Closing in..."
        },
        {
            "phase": "🎯 Refinement",
            "question": "Is there a sense of longing or wistfulness?",
            "similarity": 0.89,
            "status": "Almost there!"
        },
        {
            "phase": "✨ Convergence",
            "question": "Does this capture nostalgia with bittersweet feelings?",
            "similarity": 0.94,
            "status": "Converged! ✓"
        }
    ]

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Phase", style="cyan")
    table.add_column("Question Asked", style="white", width=50)
    table.add_column("Similarity", justify="right", style="yellow")
    table.add_column("Status", style="green")

    for step in steps:
        similarity_bar = "█" * int(step["similarity"] * 20)
        table.add_row(
            step["phase"],
            step["question"],
            f"{step['similarity']:.2f} {similarity_bar}",
            step["status"]
        )

    console.print(table)

    console.print("\n[bold green]🎉 Discovery:[/bold green] [italic]\"The bittersweet feeling of nostalgia "
                  "when looking through old photo albums\"[/italic]\n")

def demo_use_cases():
    """Show practical use cases."""

    console.print("\n[bold blue]💡 Real-World Use Cases[/bold blue]\n")

    use_cases = [
        {
            "title": "Content Discovery",
            "description": "Investigate embeddings in vector databases to generate descriptive metadata",
            "example": "Large document collections → Automatic summaries"
        },
        {
            "title": "Sentiment Analysis",
            "description": "Uncover nuanced emotions beyond positive/negative",
            "example": "\"Nostalgic longing with hints of regret\""
        },
        {
            "title": "AI Interpretability",
            "description": "Decode internal representations of AI models",
            "example": "Understanding what neural networks \"think\""
        },
        {
            "title": "Creative Exploration",
            "description": "Discover unexpected semantic connections",
            "example": "Finding themes for creative writing projects"
        }
    ]

    for i, uc in enumerate(use_cases, 1):
        console.print(f"[cyan]{i}. {uc['title']}[/cyan]")
        console.print(f"   {uc['description']}")
        console.print(f"   [dim]Example: {uc['example']}[/dim]\n")

def demo_architecture():
    """Show the architecture and key features."""

    console.print("\n[bold blue]🏗️  Architecture[/bold blue]\n")

    console.print("┌─────────────────────────────────────────┐")
    console.print("│  [yellow]1. Unknown Embedding[/yellow]             │")
    console.print("│     (vector of unknown origin)          │")
    console.print("└────────────┬────────────────────────────┘")
    console.print("             │")
    console.print("             ▼")
    console.print("┌─────────────────────────────────────────┐")
    console.print("│  [cyan]2. Question Generator (LLM)[/cyan]        │")
    console.print("│     • Exploration phase (broad)         │")
    console.print("│     • Refinement phase (narrow)         │")
    console.print("│     • Convergence detection             │")
    console.print("└────────────┬────────────────────────────┘")
    console.print("             │")
    console.print("             ▼")
    console.print("┌─────────────────────────────────────────┐")
    console.print("│  [green]3. Similarity Calculator[/green]           │")
    console.print("│     • Cosine similarity scoring         │")
    console.print("│     • \"Hot and cold\" guidance           │")
    console.print("│     • Statistical convergence           │")
    console.print("└────────────┬────────────────────────────┘")
    console.print("             │")
    console.print("             ▼")
    console.print("┌─────────────────────────────────────────┐")
    console.print("│  [magenta]4. Synthesis[/magenta]                      │")
    console.print("│     Human-readable description          │")
    console.print("│     + confidence metrics                │")
    console.print("└─────────────────────────────────────────┘\n")

def demo_quick_start():
    """Show quick start code example."""

    console.print("\n[bold blue]🚀 Quick Start Code[/bold blue]\n")

    code = """from perquire import PerquireInvestigator
from sentence_transformers import SentenceTransformer

# Initialize with Gemini (via Pydantic AI)
investigator = PerquireInvestigator(
    embedding_model=SentenceTransformer('all-MiniLM-L6-v2'),
    llm_provider='gemini'  # or 'openai', 'anthropic', 'ollama'
)

# Create a mysterious embedding
mysterious_text = "The melancholic beauty of abandoned places"
target_embedding = investigator.embedding_model.encode(mysterious_text)

# Let Perquire investigate!
result = investigator.investigate(target_embedding)

print(f"Discovery: {result.description}")
print(f"Confidence: {result.final_similarity:.1%}")
print(f"Questions asked: {result.iterations}")"""

    console.print(Panel(code, title="Python Example", border_style="green"))

def demo_features():
    """Highlight key features."""

    console.print("\n[bold blue]✨ Key Features[/bold blue]\n")

    features = [
        ("🤖 Pydantic AI Integration", "Type-safe LLM interactions with 50% code reduction"),
        ("🔄 Multi-model Support", "Gemini, OpenAI, Anthropic, Ollama"),
        ("📊 Smart Convergence", "Automatic detection when investigation is complete"),
        ("🎯 Adaptive Questioning", "Adjusts strategy based on similarity scores"),
        ("🌐 Web Interface", "FastAPI server for batch investigations"),
        ("📈 Progress Tracking", "Rich terminal output with real-time updates")
    ]

    for emoji_title, desc in features:
        console.print(f"[cyan]{emoji_title}[/cyan]")
        console.print(f"  {desc}\n")

def main():
    """Run the demo."""

    console.print("\n" + "="*60)
    console.print("[bold magenta]PERQUIRE DEMONSTRATION[/bold magenta]".center(60))
    console.print("="*60)

    demo_conceptual_explanation()

    console.print("\n[dim]Press Enter to continue...[/dim]")
    input()

    demo_investigation_process()

    console.print("\n[dim]Press Enter to continue...[/dim]")
    input()

    demo_architecture()

    console.print("\n[dim]Press Enter to continue...[/dim]")
    input()

    demo_use_cases()

    console.print("\n[dim]Press Enter to continue...[/dim]")
    input()

    demo_features()

    console.print("\n[dim]Press Enter to continue...[/dim]")
    input()

    demo_quick_start()

    console.print("\n\n[bold green]🎓 Learn More:[/bold green]")
    console.print("  • Repository: https://github.com/franklinbaldo/perquire")
    console.print("  • README: /home/frank/workspace/perquire/README.md")
    console.print("  • Examples: /home/frank/workspace/perquire/examples/")
    console.print("\n[bold cyan]To try it yourself:[/bold cyan]")
    console.print("  1. Set GOOGLE_API_KEY environment variable")
    console.print("  2. cd /home/frank/workspace/perquire")
    console.print("  3. Fix pyproject.toml (change requires-python to >=3.11)")
    console.print("  4. uv sync")
    console.print("  5. uv run python examples/live_e2e_test.py\n")

if __name__ == "__main__":
    main()
