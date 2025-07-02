#!/usr/bin/env python3
"""
Live End-to-End Test for Perquire
Generates text → Creates embedding → Investigates with CLI → Evaluates results
"""

import os
import sys
import tempfile
import subprocess
import json
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm, IntPrompt
import google.generativeai as genai

console = Console()

# Test content categories for diverse evaluation
TEST_CONTENT_CATEGORIES = [
    {
        "name": "Visual Scene", 
        "prompt": "A cozy coffee shop on a rainy evening with warm yellow lights"
    },
    {
        "name": "Abstract Emotion",
        "prompt": "The bittersweet feeling of nostalgia when looking through old photo albums"
    },
    {
        "name": "Technical Concept",
        "prompt": "Machine learning gradient descent optimization algorithms for neural networks"
    },
    {
        "name": "Creative Arts",
        "prompt": "A jazz musician improvising a saxophone solo in a dimly lit underground club"
    },
    {
        "name": "Nature Description",
        "prompt": "Ancient redwood trees towering in misty morning fog with dappled sunlight"
    }
]

def setup_api_key():
    """Check for GOOGLE_API_KEY and handle appropriately."""
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        console.print("🔑 [yellow]GOOGLE_API_KEY not found in environment[/yellow]")
        console.print("Please set your API key using:")
        console.print("export GOOGLE_API_KEY='your-api-key-here'")
        console.print("or create a .env file with GOOGLE_API_KEY=your-api-key-here")
        console.print("and run with: uv run --env-file .env python live_e2e_test.py")
        return False
    
    try:
        genai.configure(api_key=api_key)
        # Test API connection
        model = genai.GenerativeModel('gemini-2.5-flash-lite-preview-06-17')
        test_response = model.generate_content("Hello")
        console.print("✅ [green]Gemini API connection successful[/green]")
        return True
    except Exception as e:
        console.print(f"❌ [red]Failed to connect to Gemini API: {e}[/red]")
        return False

def generate_test_content():
    """Generate or select test content."""
    console.print("\n[bold blue]📝 Content Selection[/bold blue]")
    
    console.print("Choose content type:")
    table = Table()
    table.add_column("Option", style="cyan")
    table.add_column("Category", style="magenta")
    table.add_column("Content", style="white")
    
    for i, category in enumerate(TEST_CONTENT_CATEGORIES, 1):
        table.add_row(str(i), category["name"], category["prompt"])
    
    table.add_row("6", "Custom", "Enter your own text")
    
    console.print(table)
    
    choice = IntPrompt.ask("Select option", choices=[str(i) for i in range(1, 7)])
    
    if choice == 6:
        custom_text = Prompt.ask("Enter your custom text")
        return {"name": "Custom", "content": custom_text}
    else:
        category = TEST_CONTENT_CATEGORIES[choice - 1]
        return {"name": category["name"], "content": category["prompt"]}

def create_embedding(text_content):
    """Create embedding using Gemini."""
    console.print(f"\n[bold blue]🧠 Creating Embedding[/bold blue]")
    console.print(f"Text: [italic]{text_content}[/italic]")
    
    try:
        # Use text-embedding-004 model for better quality
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text_content,
            task_type="semantic_similarity"
        )
        
        embedding = np.array(result['embedding'], dtype=np.float32)
        console.print(f"✅ [green]Embedding created successfully[/green]")
        console.print(f"📊 Dimensions: {len(embedding)}")
        console.print(f"📈 Norm: {np.linalg.norm(embedding):.3f}")
        
        return embedding
        
    except Exception as e:
        console.print(f"❌ [red]Failed to create embedding: {e}[/red]")
        return None

def save_embedding_file(embedding):
    """Save embedding to temporary file."""
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.npy', delete=False) as f:
        np.save(f, embedding)
        return f.name

def run_perquire_investigation(embedding_file):
    """Run Perquire CLI investigation."""
    console.print(f"\n[bold blue]🔍 Running Perquire Investigation[/bold blue]")
    
    # Check if we have the original perquire module available
    try:
        # Try to run with the original source
        cmd = [
            sys.executable, "-m", "src.perquire.cli.main", 
            "investigate", embedding_file,
            "--llm-provider", "gemini",
            "--embedding-provider", "gemini", 
            "--verbose"
        ]
        
        console.print(f"🚀 Command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=300,  # 5 minute timeout
            cwd="/mnt/c/Users/frank/perquire"
        )
        
        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
        
    except subprocess.TimeoutExpired:
        console.print("⏰ [yellow]Investigation timed out after 5 minutes[/yellow]")
        return {"returncode": -1, "stdout": "", "stderr": "Timeout"}
    except Exception as e:
        console.print(f"❌ [red]Failed to run investigation: {e}[/red]")
        return {"returncode": -1, "stdout": "", "stderr": str(e)}

def extract_description_from_output(output):
    """Extract the final description from Perquire output."""
    lines = output.split('\n')
    
    for line in lines:
        if "Description:" in line:
            return line.split("Description:", 1)[1].strip()
        elif "Final description:" in line:
            return line.split("Final description:", 1)[1].strip()
    
    # Fallback: look for lines that seem like descriptions
    for line in lines:
        line = line.strip()
        if len(line) > 20 and not line.startswith(('✅', '❌', '🔍', '📊', '⏱️', 'Investigating')):
            return line
    
    return "No description found"

def evaluate_results(original_content, generated_description, investigation_output):
    """Subjective evaluation of results."""
    console.print(f"\n[bold green]📋 Results Evaluation[/bold green]")
    
    # Display comparison
    comparison_table = Table(title="Investigation Results")
    comparison_table.add_column("Original Content", style="cyan", width=40)
    comparison_table.add_column("Generated Description", style="magenta", width=40)
    
    comparison_table.add_row(
        original_content["content"],
        generated_description
    )
    
    console.print(comparison_table)
    
    # Show investigation details if verbose
    if investigation_output.get("stdout"):
        console.print(f"\n[bold]🔍 Investigation Output:[/bold]")
        console.print(Panel(investigation_output["stdout"], title="Perquire Output"))
    
    if investigation_output.get("stderr"):
        console.print(f"\n[bold red]⚠️ Errors/Warnings:[/bold red]")
        console.print(Panel(investigation_output["stderr"], title="Errors"))
    
    # Subjective evaluation
    console.print(f"\n[bold yellow]🎯 Subjective Evaluation[/bold yellow]")
    
    # Accuracy rating
    accuracy = IntPrompt.ask(
        "How accurately does the generated description capture the original content? (1-5)", 
        choices=["1", "2", "3", "4", "5"]
    )
    
    # Completeness rating  
    completeness = IntPrompt.ask(
        "How complete is the description? (1-5)",
        choices=["1", "2", "3", "4", "5"] 
    )
    
    # Clarity rating
    clarity = IntPrompt.ask(
        "How clear and understandable is the description? (1-5)",
        choices=["1", "2", "3", "4", "5"]
    )
    
    # Overall satisfaction
    overall = IntPrompt.ask(
        "Overall satisfaction with the investigation? (1-5)",
        choices=["1", "2", "3", "4", "5"]
    )
    
    # Comments
    comments = Prompt.ask("Any additional comments?", default="None")
    
    return {
        "accuracy": accuracy,
        "completeness": completeness, 
        "clarity": clarity,
        "overall": overall,
        "comments": comments,
        "average_score": (accuracy + completeness + clarity + overall) / 4
    }

def display_final_summary(test_session):
    """Display comprehensive test summary."""
    console.print(f"\n[bold green]📊 Test Session Summary[/bold green]")
    
    summary_table = Table(title="E2E Test Results")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="white")
    summary_table.add_column("Notes", style="dim")
    
    summary_table.add_row("Content Category", test_session["content"]["name"], "")
    summary_table.add_row("Embedding Dimensions", str(test_session["embedding_dims"]), "")
    summary_table.add_row("Investigation Status", 
                         "✅ Success" if test_session["investigation"]["returncode"] == 0 else "❌ Failed", "")
    
    if "evaluation" in test_session:
        eval_data = test_session["evaluation"]
        summary_table.add_row("Accuracy Score", f"{eval_data['accuracy']}/5", "How well it captured content")
        summary_table.add_row("Completeness Score", f"{eval_data['completeness']}/5", "How complete the description")
        summary_table.add_row("Clarity Score", f"{eval_data['clarity']}/5", "How clear the description")
        summary_table.add_row("Overall Score", f"{eval_data['overall']}/5", "Overall satisfaction")
        summary_table.add_row("Average Score", f"{eval_data['average_score']:.1f}/5", "Combined average")
    
    console.print(summary_table)
    
    # Success criteria
    if "evaluation" in test_session:
        avg_score = test_session["evaluation"]["average_score"]
        if avg_score >= 4.0:
            status = "🎉 [bold green]EXCELLENT[/bold green]"
        elif avg_score >= 3.0:
            status = "✅ [green]GOOD[/green]"
        elif avg_score >= 2.0:
            status = "⚠️ [yellow]FAIR[/yellow]"
        else:
            status = "❌ [red]NEEDS IMPROVEMENT[/red]"
        
        console.print(f"\n[bold]Test Result: {status}[/bold]")
        
        if test_session["evaluation"]["comments"] != "None":
            console.print(f"\n[bold]Comments:[/bold] {test_session['evaluation']['comments']}")

def main():
    """Main test orchestrator."""
    console.print(Panel.fit(
        "[bold blue]🧪 Perquire Live End-to-End Test[/bold blue]\n"
        "Generate text → Create embedding → Investigate → Evaluate",
        title="Live E2E Test"
    ))
    
    # Setup
    if not setup_api_key():
        return
    
    test_session = {}
    
    try:
        # Step 1: Generate/select content
        test_session["content"] = generate_test_content()
        
        # Step 2: Create embedding
        embedding = create_embedding(test_session["content"]["content"])
        if embedding is None:
            return
        
        test_session["embedding_dims"] = len(embedding)
        
        # Step 3: Save embedding to file
        embedding_file = save_embedding_file(embedding)
        console.print(f"💾 [green]Embedding saved to: {embedding_file}[/green]")
        
        # Step 4: Run Perquire investigation
        investigation_result = run_perquire_investigation(embedding_file)
        test_session["investigation"] = investigation_result
        
        if investigation_result["returncode"] == 0:
            console.print("✅ [green]Investigation completed successfully[/green]")
            
            # Step 5: Extract description
            description = extract_description_from_output(investigation_result["stdout"])
            test_session["generated_description"] = description
            
            # Step 6: Evaluate results
            evaluation = evaluate_results(
                test_session["content"], 
                description, 
                investigation_result
            )
            test_session["evaluation"] = evaluation
            
        else:
            console.print("❌ [red]Investigation failed[/red]")
            console.print(f"Error: {investigation_result['stderr']}")
        
        # Step 7: Display summary
        display_final_summary(test_session)
        
        # Ask if user wants to run another test
        if Confirm.ask("\nRun another test?"):
            main()
    
    except KeyboardInterrupt:
        console.print("\n⏹️ [yellow]Test interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n❌ [red]Test failed with error: {e}[/red]")
    finally:
        # Cleanup
        if 'embedding_file' in locals() and os.path.exists(embedding_file):
            os.unlink(embedding_file)
            console.print(f"🧹 Cleaned up temporary file: {embedding_file}")

if __name__ == "__main__":
    main()