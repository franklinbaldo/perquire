"""
Pydantic AI Provider Demo

This example demonstrates the improvements from using Pydantic AI
for LLM interactions in PERQUIRE.

Run with:
    python examples/pydantic_ai_demo.py
"""

import asyncio
import os
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from perquire.llm.pydantic_ai_provider import (
    create_gemini_provider,
    create_openai_provider,
    create_anthropic_provider
)
from perquire.llm.models import InvestigationContext

console = Console()


async def demo_question_generation():
    """Demo: Generate investigation questions with type safety."""
    console.print("\n[bold cyan]═══ QUESTION GENERATION DEMO ═══[/bold cyan]\n")

    # Create provider (model-agnostic!)
    provider = create_gemini_provider(
        model="gemini-1.5-flash",
        temperature=0.7
    )

    # Create investigation context
    context = InvestigationContext(
        current_description="Something related to emotions and human experiences",
        current_similarity=0.65,
        phase="refinement",
        previous_questions=[
            "Is this about positive or negative emotions?",
            "Does this relate to personal relationships?",
            "Is there a sense of nostalgia involved?"
        ],
        iteration=3
    )

    console.print("[yellow]Generating questions...[/yellow]")

    # Generate questions - returns validated QuestionBatch!
    try:
        batch = await provider.generate_questions(context, num_questions=3)

        # Display results
        table = Table(title="Generated Questions", show_header=True, header_style="bold magenta")
        table.add_column("Question", style="cyan", width=60)
        table.add_column("Phase", style="green")
        table.add_column("Expected Gain", justify="right", style="yellow")

        for q in batch.questions:
            table.add_row(
                q.question,
                q.phase,
                f"{q.expected_similarity_gain:.3f}"
            )

        console.print(table)
        console.print(f"\n[green]Strategy:[/green] {batch.strategy}")
        console.print(f"[green]Current Similarity:[/green] {batch.current_similarity:.3f}")

        # Show type safety in action
        console.print("\n[bold yellow]Type Safety Benefits:[/bold yellow]")
        console.print("✅ All questions are validated (10-500 chars)")
        console.print("✅ All questions end with '?'")
        console.print("✅ Phase is one of: exploration, refinement, convergence")
        console.print("✅ Expected gains are 0.0-1.0")
        console.print("✅ IDE autocomplete works perfectly!")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


async def demo_synthesis():
    """Demo: Synthesize investigation results."""
    console.print("\n[bold cyan]═══ SYNTHESIS DEMO ═══[/bold cyan]\n")

    provider = create_gemini_provider(model="gemini-1.5-flash")

    # Simulated investigation results
    results = [
        {"question": "Is this about melancholic emotions?", "similarity": 0.89},
        {"question": "Does this involve memories of the past?", "similarity": 0.87},
        {"question": "Is there a sense of beauty in sadness?", "similarity": 0.85},
        {"question": "Does this relate to abandoned or forgotten places?", "similarity": 0.83},
        {"question": "Is there a bittersweet quality to the emotion?", "similarity": 0.81},
    ]

    console.print("[yellow]Synthesizing results...[/yellow]")

    try:
        synthesis = await provider.synthesize_description(results, final_similarity=0.89)

        # Display synthesis
        console.print(Panel(
            synthesis.description,
            title="[bold green]Synthesized Description[/bold green]",
            border_style="green"
        ))

        console.print(f"\n[green]Confidence:[/green] {synthesis.confidence:.2%}")
        console.print(f"[green]Final Similarity:[/green] {synthesis.final_similarity:.3f}")

        if synthesis.key_findings:
            console.print("\n[bold]Key Findings:[/bold]")
            for finding in synthesis.key_findings:
                console.print(f"  • {finding}")

        if synthesis.uncertainty_areas:
            console.print("\n[bold]Uncertainty Areas:[/bold]")
            for area in synthesis.uncertainty_areas:
                console.print(f"  ⚠️  {area}")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


async def demo_health_check():
    """Demo: Health check across multiple providers."""
    console.print("\n[bold cyan]═══ HEALTH CHECK DEMO ═══[/bold cyan]\n")

    providers = []

    # Check which providers are available
    if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
        providers.append(("Gemini", create_gemini_provider()))

    if os.getenv("OPENAI_API_KEY"):
        providers.append(("OpenAI", create_openai_provider()))

    if os.getenv("ANTHROPIC_API_KEY"):
        providers.append(("Anthropic", create_anthropic_provider()))

    if not providers:
        console.print("[yellow]No API keys found. Set GOOGLE_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY[/yellow]")
        return

    table = Table(title="Provider Health Check", show_header=True, header_style="bold magenta")
    table.add_column("Provider", style="cyan")
    table.add_column("Model", style="green")
    table.add_column("Status", style="yellow")
    table.add_column("Response Time", justify="right", style="blue")

    for name, provider in providers:
        console.print(f"[yellow]Checking {name}...[/yellow]")
        health = await provider.health_check()

        status_emoji = "✅" if health.status == "healthy" else "❌"
        response_time = f"{health.response_time_ms:.0f}ms" if health.response_time_ms else "N/A"

        table.add_row(
            name,
            health.provider_info.model_name if health.provider_info else "N/A",
            f"{status_emoji} {health.status}",
            response_time
        )

    console.print(table)


async def demo_comparison():
    """Show before/after code comparison."""
    console.print("\n[bold cyan]═══ CODE COMPARISON ═══[/bold cyan]\n")

    before = """
# BEFORE: Manual provider (277 lines)

class GeminiProvider(BaseLLMProvider):
    def __init__(self, config):
        # 40 lines of initialization
        super().__init__(config)
        self._llm = None
        self._initialize_llm()

    def generate_questions(self, current_description, target_similarity, phase, previous_questions):
        # Build prompt manually
        prompt = self._build_question_generation_prompt(...)  # 30 lines
        response = self.generate_response(prompt)
        # Parse response manually
        questions = self._parse_questions_from_response(response.content)  # 20 lines
        return questions  # Returns List[str] - no validation!

    def _build_question_generation_prompt(self, ...):
        # 30+ lines of string concatenation
        prompt = f"You are helping investigate..."
        # ... many more lines

    def _parse_questions_from_response(self, response):
        # 20+ lines of manual parsing
        lines = response.strip().split('\\n')
        questions = []
        for line in lines:
            # ... parsing logic
        return questions
    """

    after = """
# AFTER: Pydantic AI provider (~150 lines total)

from pydantic_ai import Agent
from perquire.llm.models import QuestionBatch, InvestigationContext

provider = PydanticAIProvider("gemini-1.5-pro", temperature=0.7)

# Generate questions with full type safety
batch = await provider.generate_questions(
    context=InvestigationContext(
        current_description="...",
        current_similarity=0.65,
        phase="refinement",
        iteration=3
    ),
    num_questions=3
)

# Returns validated QuestionBatch with:
# - questions: list[InvestigationQuestion]  ✅ Validated
# - strategy: str                           ✅ Required
# - current_similarity: float               ✅ 0.0-1.0 range

# Every question is validated:
# - question: str (10-500 chars, ends with '?')
# - phase: Literal["exploration", "refinement", "convergence"]
# - expected_similarity_gain: float (0.0-1.0)
# - rationale: str
    """

    console.print(Panel(
        Markdown("## Before: Manual Provider Implementation\n```python" + before + "```"),
        border_style="red",
        title="❌ OLD WAY (277 lines)"
    ))

    console.print(Panel(
        Markdown("## After: Pydantic AI Provider\n```python" + after + "```"),
        border_style="green",
        title="✅ NEW WAY (~30 lines for usage)"
    ))

    # Show improvements
    improvements = Table(title="Improvements", show_header=True, header_style="bold magenta")
    improvements.add_column("Aspect", style="cyan")
    improvements.add_column("Before", style="red")
    improvements.add_column("After", style="green")

    improvements.add_row("Code size", "277 lines/provider", "~150 lines total")
    improvements.add_row("Type safety", "❌ None", "✅ Full Pydantic validation")
    improvements.add_row("Manual parsing", "✅ Required", "❌ Automatic")
    improvements.add_row("Validation", "❌ Manual", "✅ Automatic")
    improvements.add_row("Provider switching", "❌ New class each", "✅ One parameter")
    improvements.add_row("Observability", "❌ None", "✅ Logfire integration")
    improvements.add_row("IDE support", "⚠️ Limited", "✅ Full autocomplete")

    console.print("\n")
    console.print(improvements)


async def main():
    """Run all demos."""
    console.print(Panel.fit(
        "[bold yellow]PERQUIRE + Pydantic AI Demo[/bold yellow]\n"
        "Demonstrating improved LLM interactions with type safety",
        border_style="yellow"
    ))

    # Check for API key
    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        console.print("\n[red]⚠️  GOOGLE_API_KEY or GEMINI_API_KEY not set![/red]")
        console.print("[yellow]Some demos will be skipped.[/yellow]\n")

    # Run demos
    await demo_comparison()

    if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
        await demo_question_generation()
        await demo_synthesis()
        await demo_health_check()
    else:
        console.print("\n[yellow]Skipping live demos (no API key)[/yellow]")

    console.print("\n[bold green]✅ Demo complete![/bold green]")
    console.print("\n[dim]To run with live API calls, set GOOGLE_API_KEY or GEMINI_API_KEY[/dim]")


if __name__ == "__main__":
    asyncio.run(main())
