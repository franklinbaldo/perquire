import click
from sentence_transformers import SentenceTransformer

from perquire.core.investigator import PerquireInvestigator
from perquire.llm.openai_provider import OpenAIProvider # Assuming default LLM
from perquire.embeddings.base import EmbeddingProvider

class DemoEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def get_embedding(self, text: str):
        return self.model.encode(text)

    def get_embedding_model_name(self) -> str:
        return self.model._first_module().name # type: ignore

@click.command("text")
@click.option(
    "--text",
    required=True,
    help="The text to investigate."
)
@click.option(
    "--llm-provider",
    default="openai",
    type=click.Choice(["openai", "anthropic", "ollama", "gemini"]), # Add other providers as needed
    help="LLM provider to use for the investigation."
)
@click.option(
    "--llm-model",
    default=None, # Provider will use its default
    help="Specific LLM model to use (optional, depends on provider)."
)
def text_demo(text: str, llm_provider: str, llm_model: str | None):
    """
    Investigates a given text using Perquire with an in-memory database.
    """
    click.echo(f"Investigating text: \"{text}\"")
    click.echo(f"Using LLM provider: {llm_provider}" + (f" with model: {llm_model}" if llm_model else ""))

    try:
        embedding_provider = DemoEmbeddingProvider()
        target_embedding = embedding_provider.get_embedding(text)

        # Configure LLM provider based on choice
        # This part might need more robust provider selection/configuration
        if llm_provider == "openai":
            llm_service = OpenAIProvider(model=llm_model if llm_model else "gpt-3.5-turbo")
        # Add other providers here once their setup is clear from the main code
        # elif llm_provider == "anthropic":
        #     llm_service = AnthropicProvider(model=llm_model if llm_model else "claude-2")
        else:
            click.echo(f"LLM provider '{llm_provider}' is not fully configured in this demo yet. Defaulting to OpenAI.", err=True)
            llm_service = OpenAIProvider(model=llm_model if llm_model else "gpt-3.5-turbo")


        investigator = PerquireInvestigator(
            embedding_provider=embedding_provider,
            llm_provider=llm_service, # type: ignore
            db_path=":memory:",  # Use in-memory DuckDB
            verbose=True # Provide some output during investigation
        )

        result = investigator.investigate(target_embedding)

        click.echo("\n--- Investigation Complete ---")
        click.echo(f"Discovered Description: {result.description}")
        click.echo(f"Final Similarity Score: {result.final_similarity:.4f}")
        click.echo(f"Iterations: {result.iterations}")
        if result.confidence_score:
            click.echo(f"Confidence Score: {result.confidence_score:.4f}")

    except Exception as e:
        click.echo(f"An error occurred: {e}", err=True)
