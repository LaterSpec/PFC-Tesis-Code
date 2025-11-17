"""Interactive demo for the inspect_schema module.

This script demonstrates the full inspect_schema pipeline with detailed output
for each step, allowing you to see exactly what happens during schema inspection.

Usage:
    python demo_inspect_schema.py
    python demo_inspect_schema.py --question "Your custom question here"
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import json
from src.modules.inspect_schema import inspect_schema


def print_section(title: str, level: int = 1):
    """Print a formatted section header."""
    if level == 1:
        print(f"\n{'=' * 70}")
        print(f"  {title}")
        print(f"{'=' * 70}\n")
    elif level == 2:
        print(f"\n[{title}]")
        print("-" * 70)


def print_success(message: str):
    """Print a success message."""
    print(f"✓ {message}")


def print_info(message: str, indent: int = 0):
    """Print an info message."""
    prefix = "  " * indent
    print(f"{prefix}• {message}")


def print_json(data: dict, indent: int = 2):
    """Print JSON data in a readable format."""
    print(json.dumps(data, indent=indent, default=str))


def run_demo(question: str, use_cache: bool = True, verbose: bool = True, use_llm: bool = False):
    """Run the inspect_schema demo with detailed output."""
    
    print_section("DEMO: inspect_schema Pipeline", level=1)
    
    # Display the question
    print(f"Question: {question}\n")
    
    # Configure database connection
    db_config = {
        "engine": "bigquery",
        "project_id": "bigquery-public-data",
        "datasets": ["usa_names"],
        "credential_path": "src/cred/clean_node.json",
        "use_cache": use_cache,
        "cache_path": "link/cache/schema_usa_names.json"
    }
    
    # Initialize LLM pipeline if requested
    llm_pipeline = None
    tokenizer = None
    if use_llm:
        print_section("Loading LLM", level=2)
        try:
            from transformers import AutoTokenizer, pipeline
            import torch
            
            #model_id = "cycloneboy/CscSQL-Merge-Qwen2.5-Coder-1.5B-Instruct"
            model_id = "meta-llama/Llama-3.1-8B-Instruct"
            device = 0 if torch.cuda.is_available() else -1
            device_name = "GPU" if device == 0 else "CPU"
            
            print_info(f"Loading {model_id}...")
            print_info(f"Device: {device_name}")
            
            # Load tokenizer first
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            
            # Create pipeline with tokenizer
            llm_pipeline = pipeline(
                "text-generation",
                model=model_id,
                tokenizer=tokenizer,
                device=device
            )
            print_success("LLM loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load LLM: {e}")
            print_info("Falling back to heuristic mode")
            use_llm = False
    
    # Configure inspection parameters
    inspect_config = {
        "max_candidates": 5,
        "max_final_schemas": 1,
        "serialization": {
            "style": "compact",
            "include_types": True,
            "include_descriptions": True
        },
        "extraction": {
            "mode": "llm" if use_llm else "heuristic",
            "llm_max_new_tokens": 128,
            "llm_temperature": 0.0,
        },
        "llm_pipeline": llm_pipeline,  # Pass the pipeline instance
        "llm_tokenizer": tokenizer      # Pass the tokenizer for chat template
    }
    
    print_section("Configuration", level=2)
    print_info(f"Engine: {db_config['engine']}")
    print_info(f"Dataset: {db_config['project_id']}.{db_config['datasets'][0]}")
    print_info(f"Use cache: {use_cache}")
    print_info(f"Extraction mode: {inspect_config['extraction']['mode']}")
    
    # Run the pipeline
    print_section("Pipeline Execution", level=2)
    
    try:
        print_info("Step 1/6: Loading metadata...")
        result = inspect_schema(
            question=question,
            db_config=db_config,
            inspect_config=inspect_config
        )
        
        # Extract results
        schema_context = result["schema_context"]
        schema_signals = result["schema_signals"]
        debug_info = result["debug_info"]
        
        # Display metadata loading results
        retrieval = debug_info["retrieval_result"]
        print_success(f"Found {len(retrieval.candidates)} database(s)")
        for cand in retrieval.candidates:
            print_info(f"{cand['db_id']} (score: {cand['score']})", indent=1)
        
        # Display retrieval results
        print("\n")
        print_info("Step 2/6: Retrieving candidate schemas...")
        print_success(f"Retrieved {len(retrieval.candidates)} schema(s)")
        print_info(f"Retrieval rounds: {retrieval.round_id + 1}", indent=1)
        
        # Display filtering results
        print("\n")
        print_info("Step 3/6: Filtering irrelevant schemas...")
        filter_result = debug_info["filter_result"]
        print_success(f"Kept {filter_result.stats['num_kept']} schema(s)")
        print_info(f"Removed: {filter_result.stats['num_removed']}", indent=1)
        print_info(f"Fraction removed: {filter_result.stats['fraction_removed']:.1%}", indent=1)
        
        # Display extraction results
        print("\n")
        print_info("Step 4/6: Extracting relevant columns...")
        extraction = debug_info["extraction_result"]
        selected_schema = schema_context["selected_schema"]
        
        total_cols = extraction.coverage_stats["num_columns_total"]
        needed_cols = extraction.coverage_stats["num_columns_needed"]
        print_success(f"Selected {needed_cols}/{total_cols} columns as needed")
        
        # Display column relevance details
        if selected_schema.tables:
            table = selected_schema.tables[0]
            print_info("Column relevance:", indent=1)
            for col in table.columns:
                is_needed = col.extra.get("needed", False)
                status = "✓" if is_needed else "✗"
                print(f"      {status} {col.name} ({col.type}): {is_needed}")
        
        # Display serialization results
        print("\n")
        print_info("Step 5/6: Serializing schema context...")
        print_success(f"Generated schema text ({schema_context['tokens_estimate']} tokens)")
        
        if verbose:
            print("\n--- Schema Text ---")
            print(schema_context["schema_text"])
            print("--- End Schema Text ---\n")
        
        # Display signals
        print_info("Step 6/6: Building signals for SCM...")
        print_success("Generated schema signals")
        
        print_section("Results Summary", level=2)
        
        print("Schema Signals:")
        for key, value in schema_signals.items():
            if "fraction" in key:
                print_info(f"{key}: {value:.1%}")
            else:
                print_info(f"{key}: {value}")
        
        print("\nSchema Context:")
        print_info(f"Database: {selected_schema.db_id}")
        print_info(f"Engine: {selected_schema.engine}")
        print_info(f"Tables: {len(selected_schema.tables)}")
        print_info(f"Total columns: {total_cols}")
        print_info(f"Needed columns: {needed_cols}")
        print_info(f"Token estimate: {schema_context['tokens_estimate']}")
        
        print_section("Pipeline Completed Successfully!", level=1)
        
        return result
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        if verbose:
            print("\nFull traceback:")
            traceback.print_exc()
        return None


def run_interactive_mode(use_llm: bool = False):
    """Run in interactive mode, asking for questions."""
    print_section("Interactive Mode", level=1)
    print("Enter questions to analyze (type 'quit' to exit)")
    print("Example: How many babies named Anthony were born in Texas in 1990?\n")
    
    while True:
        try:
            question = input("\nQuestion: ").strip()
            
            if question.lower() in ["quit", "exit", "q"]:
                print("\nGoodbye!")
                break
            
            if not question:
                print("Please enter a question.")
                continue
            
            run_demo(question, use_cache=True, verbose=False, use_llm=use_llm)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Demo the inspect_schema pipeline"
    )
    parser.add_argument(
        "--question",
        type=str,
        default="How many babies named Anthony were born in Texas in 1990?",
        help="Question to analyze"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable schema caching (queries BigQuery directly)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Use LLM for column extraction (default: heuristic)"
    )
    
    args = parser.parse_args()
    
    if args.interactive:
        run_interactive_mode(use_llm=args.llm)
    else:
        run_demo(
            question=args.question,
            use_cache=not args.no_cache,
            verbose=not args.quiet,
            use_llm=args.llm
        )


if __name__ == "__main__":
    main()
