import os
import sys
import subprocess
import numpy as np
from pathlib import Path
import tempfile
import json # For parsing LLM response for evaluation

# Attempt to import google.generativeai and handle if not found initially
try:
    import google.generativeai as genai
except ImportError:
    print("🔴 ERROR: google-generativeai library not found. Please install it (e.g., pip install google-generativeai)")
    print("         This library is required for LLM-based evaluation in the E2E test.")
    sys.exit(1)

# --- Perquire Imports ---
try:
    from perquire.embeddings import embedding_registry, GeminiEmbeddingProvider
    from perquire.exceptions import EmbeddingError, ConfigurationError
except ImportError as e:
    print(f"🔴 ERROR: Failed to import Perquire components. Ensure Perquire is installed correctly. Details: {e}")
    sys.exit(1)

# --- Configuration ---
SOURCE_TEXT_DIR = Path(__file__).parent.parent / "e2e_data" / "source_texts"
SOURCE_TEXT_FILE = SOURCE_TEXT_DIR / "sample1.txt"

TEMP_EMBEDDING_DIR_OBJ = tempfile.TemporaryDirectory(prefix="perquire_e2e_")
TEMP_EMBEDDING_DIR = Path(TEMP_EMBEDDING_DIR_OBJ.name)
GENERATED_EMBEDDING_FILE = TEMP_EMBEDDING_DIR / "generated_sample1.npy"

PERQUIRE_CLI_COMMAND = "perquire"
MIN_ACCEPTABLE_EVALUATION_RATING = 3 # Minimum LLM rating (e.g., out of 5) to pass the test

def generate_embedding_from_text(text_file_path: Path, output_embedding_path: Path, api_key: str) -> bool:
    print(f"\n--- Generating embedding for: {text_file_path} ---")
    try:
        if not text_file_path.exists():
            print(f"🔴 ERROR: Source text file not found at {text_file_path}")
            return False

        with open(text_file_path, 'r') as f:
            text_content = f.read()

        if not text_content.strip():
            print("🔴 ERROR: Source text file is empty.")
            return False

        gemini_config = {"api_key": api_key, "model": "models/embedding-001"}
        try:
            try:
                gemini_provider = embedding_registry.get_provider("gemini")
                if not gemini_provider.config.get("api_key") and not os.getenv("GEMINI_API_KEY"):
                     gemini_provider.config.update(gemini_config)
                     if hasattr(gemini_provider, "_initialize_model"): gemini_provider._initialize_model()
                if not gemini_provider.is_available():
                    print("Retrying Gemini provider initialization with direct config for E2E.")
                    gemini_provider = GeminiEmbeddingProvider(config=gemini_config)
            except Exception:
                print("Failed to get Gemini provider from registry, creating new instance for E2E.")
                gemini_provider = GeminiEmbeddingProvider(config=gemini_config)
        except ConfigurationError as ce:
            print(f"🔴 ERROR: Configuration error initializing Gemini provider: {ce}")
            return False

        print(f"Using Gemini embedding model: {gemini_provider.get_model_info().get('model')}")
        embedding_result = gemini_provider.embed_text(text_content)
        output_embedding_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_embedding_path, embedding_result.embedding)
        print(f"✅ Embedding generated and saved to: {output_embedding_path}")
        print(f"    Embedding shape: {embedding_result.embedding.shape}, dtype: {embedding_result.embedding.dtype}")
        return True
    except EmbeddingError as ee:
        print(f"🔴 ERROR: Embedding generation failed: {ee}")
        return False
    except Exception as e:
        print(f"🔴 ERROR: An unexpected error occurred during embedding generation: {e}")
        return False

def evaluate_description_with_llm(original_text: str, generated_description: str, api_key: str) -> dict:
    """
    Uses Gemini to evaluate how well the generated_description matches the original_text.
    Returns a dictionary with 'rating' (int), 'justification' (str), and 'error' (str or None).
    """
    print("\n--- Evaluating description with LLM ---")
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro') # Or another suitable model

        prompt = f"""
        Original Text:
        \"\"\"
        {original_text}
        \"\"\"

        Generated Description:
        \"\"\"
        {generated_description}
        \"\"\"

        Based on the Original Text, how accurately and comprehensively does the Generated Description
        capture its key elements? Provide a rating from 1 to 5 (1=Poor, 5=Excellent) and a brief
        justification for your rating.

        Respond ONLY in JSON format with keys "rating" (an integer) and "justification" (a string).
        Example: {{"rating": 4, "justification": "The description captures most key elements but misses a minor detail."}}
        """

        # print(f"Evaluation Prompt:\n{prompt}") # For debugging
        response = model.generate_content(prompt)

        # print(f"LLM Evaluation Raw Response Text:\n{response.text}") # For debugging

        # Attempt to parse the JSON response
        # Gemini might wrap JSON in ```json ... ```, so try to extract it.
        cleaned_response_text = response.text.strip()
        if cleaned_response_text.startswith("```json"):
            cleaned_response_text = cleaned_response_text[7:]
        if cleaned_response_text.endswith("```"):
            cleaned_response_text = cleaned_response_text[:-3]

        eval_result = json.loads(cleaned_response_text)

        if not isinstance(eval_result.get("rating"), int) or not (1 <= eval_result.get("rating") <= 5):
            raise ValueError("LLM response 'rating' is missing, not an int, or out of range (1-5).")
        if not isinstance(eval_result.get("justification"), str) or not eval_result.get("justification").strip():
            raise ValueError("LLM response 'justification' is missing or empty.")

        return {
            "rating": eval_result["rating"],
            "justification": eval_result["justification"],
            "error": None
        }
    except json.JSONDecodeError as jde:
        error_msg = f"LLM evaluation response was not valid JSON. Response: '{response.text}'. Error: {jde}"
        print(f"🔴 {error_msg}")
        return {"rating": 0, "justification": "", "error": error_msg}
    except ValueError as ve: # Catch our custom validation errors
        error_msg = f"LLM evaluation response malformed. Error: {ve}. Response: '{response.text}'"
        print(f"🔴 {error_msg}")
        return {"rating": 0, "justification": "", "error": error_msg}
    except Exception as e:
        error_msg = f"An unexpected error occurred during LLM evaluation: {e}. Response: '{getattr(response, 'text', 'N/A')}'"
        print(f"🔴 {error_msg}")
        return {"rating": 0, "justification": "", "error": error_msg}


def main():
    print("--- Starting Perquire E2E Test (with LLM Evaluation) ---")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("🔴 ERROR: GEMINI_API_KEY environment variable not set.")
        sys.exit(1)
    print("✅ GEMINI_API_KEY found.")

    if not generate_embedding_from_text(SOURCE_TEXT_FILE, GENERATED_EMBEDDING_FILE, api_key):
        print("🔴 TEST FAILED: Could not generate embedding from source text.")
        sys.exit(1)

    try:
        with open(SOURCE_TEXT_FILE, 'r') as f:
            original_text_content = f.read().strip()
        print(f"\n📜 Original Text Content (for reference):\n---\n{original_text_content}\n---")
    except Exception as e:
        print(f"🔴 ERROR: Could not read source text file for comparison: {e}")
        sys.exit(1)

    investigate_command_args = [
        PERQUIRE_CLI_COMMAND, "investigate", str(GENERATED_EMBEDDING_FILE),
        "--llm-provider", "gemini", "--embedding-provider", "gemini", "--verbose"
    ]
    print(f"\n🚀 Running investigation command: {' '.join(investigate_command_args)}")

    investigation_passed = False
    generated_description = ""
    try:
        process = subprocess.run(investigate_command_args, capture_output=True, text=True, check=False, timeout=450)
        print("\n--- Investigation Command Output ---")
        print(process.stdout)
        if process.stderr: print(f"\n--- Investigation Command Errors ---\n{process.stderr}")
        print("--- End of Investigation Output ---")

        if process.returncode != 0:
            print(f"🔴 INVESTIGATION FAILED: Command exited with error code {process.returncode}")
            sys.exit(1)

        description_marker = "Description:"
        final_desc_marker = "Final description:"
        lines = process.stdout.splitlines()
        for line in lines:
            if description_marker in line:
                generated_description = line.split(description_marker, 1)[1].strip()
                break
            elif final_desc_marker in line and not generated_description: # Fallback
                 generated_description = line.split(final_desc_marker, 1)[1].strip()
                 break

        print(f"\n💬 Generated Description:\n---\n{generated_description}\n---")
        if not generated_description or len(generated_description) < 10:
            print("🔴 TEST FAILED: Generated description is missing or too short.")
            sys.exit(1)

        print("✅ INVESTIGATION SUCCEEDED: Command ran successfully and a description was generated.")
        investigation_passed = True
    except subprocess.TimeoutExpired:
        print("🔴 TEST FAILED: Investigation command timed out.")
        sys.exit(1)
    except FileNotFoundError:
        print(f"🔴 TEST FAILED: '{PERQUIRE_CLI_COMMAND}' command not found.")
        sys.exit(1)
    except Exception as e:
        print(f"🔴 TEST FAILED: An unexpected error occurred during investigation: {e}")
        sys.exit(1)

    if investigation_passed:
        evaluation = evaluate_description_with_llm(original_text_content, generated_description, api_key)
        print(f"\n📊 LLM Evaluation Result:")
        print(f"   Rating (1-5): {evaluation['rating']}")
        print(f"   Justification: {evaluation['justification']}")
        if evaluation['error']:
            print(f"   Evaluation Error: {evaluation['error']}")
            print("🔴 TEST FAILED: LLM-based evaluation encountered an error.")
            sys.exit(1)

        if evaluation['rating'] < MIN_ACCEPTABLE_EVALUATION_RATING:
            print(f"🔴 TEST FAILED: LLM evaluation rating {evaluation['rating']} is below threshold of {MIN_ACCEPTABLE_EVALUATION_RATING}.")
            sys.exit(1)
        else:
            print(f"✅ LLM Evaluation PASSED: Rating {evaluation['rating']} is acceptable.")

    if investigation_passed: # Double check, though sys.exit would have occurred
        print("\n--- Perquire E2E Test Completed Successfully (including LLM evaluation) ---")
    else:
        print("\n--- Perquire E2E Test FAILED (Reason should be logged above) ---")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    finally:
        try:
            TEMP_EMBEDDING_DIR_OBJ.cleanup()
            print(f"\n🧹 Cleaned up temporary directory: {TEMP_EMBEDDING_DIR_OBJ.name}")
        except Exception as e:
            print(f"⚠️ Warning: Could not cleanup temporary directory {TEMP_EMBEDDING_DIR_OBJ.name}: {e}")
