"""Interactive demo for inspect_schema with Snowflake backend.

This script demonstrates the full inspect_schema pipeline using Snowflake
instead of BigQuery, using credentials from JSON file.

Usage:
    python link/demo_inspect_schema_snowflake.py
    python link/demo_inspect_schema_snowflake.py --llm
    python link/demo_inspect_schema_snowflake.py --no-cache
    python link/demo_inspect_schema_snowflake.py --question "Custom question"
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
    """Run the inspect_schema demo with Snowflake backend."""
    
    print_section("DEMO: inspect_schema Pipeline (Snowflake)", level=1)
    
    # Display the question
    print(f"Question: {question}\n")
    
    # Configure Snowflake connection using JSON credentials
    db_config = {
        "engine": "snowflake",
        "credential_path": "src/cred/snowflake_credential.json",
        "databases": ["USA_NAMES"],
        "schemas": ["USA_NAMES"],
        "use_cache": use_cache,
        "cache_path": "link/cache/schema_usa_names_snowflake.json",
    }
    
    # Initialize LLM pipeline if requested
    llm_pipeline = None
    tokenizer = None
    if use_llm:
        print_section("Loading LLM", level=2)
        try:
            from transformers import AutoTokenizer, pipeline
            import torch
            
            model_id = "meta-llama/Llama-3.2-3B-Instruct"
            print_info(f"Loading {model_id}...")
            
            # Load tokenizer first
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            
            # Check for GPU
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print_info(f"Device: {'GPU' if device == 'cuda' else 'CPU'}")
            
            # Create pipeline
            llm_pipeline = pipeline(
                "text-generation",
                model=model_id,
                tokenizer=tokenizer,
                device=0 if device == "cuda" else -1,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            )
            
            print_success("LLM loaded successfully")
        except Exception as e:
            print(f"[ERROR] Failed to load LLM: {e}")
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
            "llm_max_new_tokens": 256,
            "llm_temperature": 0.0,
            "llm_do_sample": False,
            "llm_repetition_penalty": 1.0,
        },
        "llm_pipeline": llm_pipeline,
        "llm_tokenizer": tokenizer
    }
    
    print_section("Configuration", level=2)
    print_info(f"Engine: {db_config['engine']}")
    print_info(f"Database: {db_config['databases']}")
    print_info(f"Schema: {db_config['schemas']}")
    print_info(f"Use cache: {use_cache}")
    print_info(f"Extraction mode: {inspect_config['extraction']['mode']}")
    
    # Run the pipeline
    print_section("Pipeline Execution", level=2)
    
    try:
        result = inspect_schema(question, db_config, inspect_config)
        
        # Extract results
        schema_context = result["schema_context"]
        schema_signals = result["schema_signals"]
        debug_info = result.get("debug_info", {})
        
        # Display step-by-step progress
        print_success(f"Found {schema_signals.num_db_initial} database(s)")
        print_info(f"{schema_context.db_id} (score: {schema_signals.avg_schema_score})", indent=1)
        
        print()
        print_success(f"Retrieved {schema_signals.num_retrieval_rounds} schema(s)")
        print_info(f"Retrieval rounds: {schema_signals.num_retrieval_rounds}", indent=1)
        
        print()
        print_success(f"Kept {schema_signals.num_db_kept} schema(s)")
        print_info(f"Removed: {schema_signals.num_db_initial - schema_signals.num_db_kept}", indent=1)
        print_info(f"Fraction removed: {schema_signals.fraction_db_removed:.1%}", indent=1)
        
        print()
        needed_cols = int(schema_context.needed_columns) if hasattr(schema_context, 'needed_columns') else 0
        total_cols = int(schema_signals.num_columns_total)
        print_success(f"Selected {needed_cols}/{total_cols} columns as needed")
        
        print()
        token_estimate = schema_context.extra.get("token_estimate", 0)
        print_success(f"Generated schema text ({token_estimate} tokens)")
        
        print()
        print_success("Generated schema signals")
        
        # Display schema text
        print_section("Schema Context", level=2)
        print("--- Schema Text ---")
        print(schema_context.schema_text)
        print("--- End Schema Text ---")
        
        # Display detailed table info if verbose
        if verbose and schema_context.tables:
            print_section("Detailed Schema Information", level=2)
            for table in schema_context.tables:
                print(f"\n• Table: {table.full_name}")
                needed_in_table = [c for c in table.columns if c.extra.get("needed", False)]
                print(f"  Columns: {len(table.columns)} total, {len(needed_in_table)} needed")
                
                if needed_in_table:
                    print("  Needed columns:")
                    for col in needed_in_table:
                        reason = col.extra.get("reason", "No reason provided")
                        print(f"    - {col.name} ({col.type}): {reason}")
        
        # Display signals
        print_section("Results Summary", level=2)
        print("Schema Signals:")
        print_info(f"num_retrieval_rounds: {schema_signals.num_retrieval_rounds}")
        print_info(f"num_db_initial: {schema_signals.num_db_initial}")
        print_info(f"num_db_kept: {schema_signals.num_db_kept}")
        print_info(f"fraction_db_removed: {schema_signals.fraction_db_removed:.1%}")
        print_info(f"num_columns_total: {schema_signals.num_columns_total}")
        print_info(f"num_columns_selected: {schema_signals.num_columns_selected}")
        print_info(f"fraction_columns_selected: {schema_signals.fraction_columns_selected:.1%}")
        print_info(f"avg_schema_score: {schema_signals.avg_schema_score}")
        
        print()
        print("Schema Context:")
        print_info(f"Database: {schema_context.db_id}")
        print_info(f"Engine: {schema_context.engine}")
        print_info(f"Tables: {len(schema_context.tables)}")
        print_info(f"Total columns: {total_cols}")
        print_info(f"Needed columns: {needed_cols}")
        print_info(f"Token estimate: {token_estimate}")
        
        print_section("Pipeline Completed Successfully!", level=1)
        
        return result
        
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Demo for inspect_schema module with Snowflake backend"
    )
    parser.add_argument(
        "--question",
        type=str,
        default="How many babies named Anthony were born in Texas in 1990?",
        help="Natural language question to process"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable schema caching (always fetch fresh from Snowflake)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Use LLM for column extraction instead of heuristics"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode (ask multiple questions)"
    )
    
    args = parser.parse_args()
    
    if args.interactive:
        print("=== Interactive Mode ===")
        print("Enter questions (or 'quit' to exit):\n")
        
        while True:
            try:
                question = input("Question: ").strip()
                if question.lower() in ["quit", "exit", "q"]:
                    break
                if not question:
                    continue
                
                run_demo(
                    question=question,
                    use_cache=not args.no_cache,
                    verbose=not args.quiet,
                    use_llm=args.llm
                )
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
    else:
        run_demo(
            question=args.question,
            use_cache=not args.no_cache,
            verbose=not args.quiet,
            use_llm=args.llm
        )


if __name__ == "__main__":
    main()
