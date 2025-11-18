"""
Demo script for gen_sql module with Snowflake backend.

This script demonstrates the full pipeline:
1. inspect_schema: Load schema and select relevant columns
2. gen_sql: Generate SQL query using LLM

Usage:
    python link/demo_gen_sql_snowflake.py
    python link/demo_gen_sql_snowflake.py --llm llama
    python link/demo_gen_sql_snowflake.py --question "Your question"
    python link/demo_gen_sql_snowflake.py --interactive
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import json
from src.modules.inspect_schema import inspect_schema
from src.modules.gen_sql import gen_sql


def print_section(title: str, level: int = 1):
    """Print formatted section header."""
    if level == 1:
        print(f"\n{'=' * 70}")
        print(f"  {title}")
        print(f"{'=' * 70}\n")
    elif level == 2:
        print(f"\n[{title}]")
        print("-" * 70)


def print_success(message: str):
    """Print success message."""
    print(f"✓ {message}")


def print_info(message: str, indent: int = 0):
    """Print info message."""
    prefix = "  " * indent
    print(f"{prefix}• {message}")


def load_llm(model_name: str = "llama"):
    """
    Load LLM model and tokenizer.
    
    Args:
        model_name: "llama" or "qwen"
        
    Returns:
        Tuple of (pipeline, tokenizer)
    """
    from transformers import AutoTokenizer, pipeline
    import torch
    
    model_map = {
        "llama": "meta-llama/Llama-3.2-3B-Instruct",
        "qwen": "cycloneboy/CscSQL-Merge-Qwen2.5-Coder-1.5B-Instruct",
    }
    
    model_id = model_map.get(model_name, model_map["llama"])
    
    print_info(f"Loading {model_id}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    # Check GPU
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
    
    return llm_pipeline, tokenizer


def run_demo(question: str, use_cache: bool = True, model_name: str = "llama"):
    """Run the full pipeline demo."""
    
    print_section("DEMO: inspect_schema + gen_sql Pipeline (Snowflake)", level=1)
    print(f"Question: {question}\n")
    
    # === STEP 1: Load LLM ===
    print_section("Step 1: Loading LLM", level=2)
    try:
        llm_pipeline, tokenizer = load_llm(model_name)
    except Exception as e:
        print(f"[ERROR] Failed to load LLM: {e}")
        return None
    
    # === STEP 2: Configure Snowflake connection ===
    print_section("Step 2: Configure Database", level=2)
    
    db_config = {
        "engine": "snowflake",
        "credential_path": "src/cred/snowflake_credential.json",
        "databases": ["USA_NAMES"],
        "schemas": ["USA_NAMES"],
        "use_cache": use_cache,
        "cache_path": "link/cache/schema_usa_names_snowflake.json",
    }
    
    print_info(f"Engine: {db_config['engine']}")
    print_info(f"Database: {db_config['databases']}")
    print_info(f"Schema: {db_config['schemas']}")
    print_info(f"Use cache: {use_cache}")
    
    # === STEP 3: Run inspect_schema ===
    print_section("Step 3: Inspect Schema (Column Selection)", level=2)
    
    inspect_config = {
        "max_candidates": 5,
        "max_final_schemas": 1,
        "serialization": {
            "style": "compact",
            "include_types": True,
            "include_descriptions": True
        },
        "extraction": {
            "mode": "llm",
            "llm_max_new_tokens": 256,
            "llm_temperature": 0.0,
            "llm_do_sample": False,
            "llm_repetition_penalty": 1.0,
        },
        "llm_pipeline": llm_pipeline,
        "llm_tokenizer": tokenizer
    }
    
    try:
        inspect_result = inspect_schema(question, db_config, inspect_config)
        
        schema_context = inspect_result["schema_context"]
        schema_signals = inspect_result["schema_signals"]
        selected_schema = schema_context["selected_schema"]
        
        print_success(f"Schema inspection completed")
        print_info(f"Database: {selected_schema.db_id}")
        print_info(f"Tables: {len(selected_schema.tables)}")
        print_info(f"Selected columns: {int(schema_signals['num_columns_selected'])}/{int(schema_signals['num_columns_total'])}")
        
        print("\n--- Schema Text ---")
        print(schema_context["schema_text"])
        print("--- End Schema Text ---")
        
    except Exception as e:
        print(f"[ERROR] inspect_schema failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # === STEP 4: Run gen_sql ===
    print_section("Step 4: Generate SQL Query", level=2)
    
    gen_config = {
        "llm_pipeline": llm_pipeline,
        "llm_tokenizer": tokenizer,
        "num_candidates": 1,
        "generation": {
            "max_new_tokens": 512,
            "temperature": 0.0,
            "do_sample": False,
            "repetition_penalty": 1.0,
        }
    }
    
    try:
        gen_result = gen_sql(question, schema_context, gen_config)
        
        sql = gen_result["sql"]
        expected_shape = gen_result["expected_shape"]
        gen_signals = gen_result["gen_signals"]
        
        print_success("SQL generation completed")
        
        print("\n--- Generated SQL ---")
        print(sql)
        print("--- End SQL ---")
        
        print("\n--- Expected Result Shape ---")
        print(f"  Kind: {expected_shape.get('kind', 'unknown')}")
        print(f"  Rows: {expected_shape.get('rows', 'unknown')}")
        
        print("\n--- Validation Status ---")
        print_info(f"Format OK: {bool(gen_signals.get('primary_format_ok', 0))}")
        print_info(f"Errors: {int(gen_signals.get('num_errors', 0))}")
        print_info(f"Warnings: {int(gen_signals.get('num_warnings', 0))}")
        print_info(f"Risk Score: {gen_signals.get('risk_score', 0.0):.2f}")
        print_info(f"Confidence Score: {gen_signals.get('confidence_score', 1.0):.2f}")
        
        print("\n--- SQL Features ---")
        print_info(f"Has aggregation: {bool(gen_signals.get('has_agg_func', 0))}")
        print_info(f"Has GROUP BY: {bool(gen_signals.get('has_group_by', 0))}")
        print_info(f"Has WHERE: {bool(gen_signals.get('has_where', 0))}")
        print_info(f"Has LIMIT: {bool(gen_signals.get('has_limit', 0))}")
        print_info(f"Has JOIN: {bool(gen_signals.get('has_join', 0))}")
        print_info(f"Length (tokens): {int(gen_signals.get('primary_sql_length_tokens', 0))}")
        
    except Exception as e:
        print(f"[ERROR] gen_sql failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # === STEP 5: Display combined signals ===
    print_section("Step 5: Combined Signals (for SCM)", level=2)
    
    print("Schema Signals:")
    print_info(f"num_columns_selected: {schema_signals.get('num_columns_selected', 0)}")
    print_info(f"fraction_columns_selected: {schema_signals.get('fraction_columns_selected', 0):.2%}")
    print_info(f"avg_schema_score: {schema_signals.get('avg_schema_score', 0):.2f}")
    
    print("\nGeneration Signals:")
    print_info(f"primary_format_ok: {gen_signals.get('primary_format_ok', 0)}")
    print_info(f"risk_score: {gen_signals.get('risk_score', 0):.2f}")
    print_info(f"confidence_score: {gen_signals.get('confidence_score', 0):.2f}")
    
    print_section("Pipeline Completed Successfully!", level=1)
    
    return {
        "question": question,
        "schema_context": schema_context,
        "schema_signals": schema_signals,
        "sql": sql,
        "expected_shape": expected_shape,
        "gen_signals": gen_signals,
    }


def main():
    """Main entry point with CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Demo for inspect_schema + gen_sql with Snowflake"
    )
    parser.add_argument(
        "--question",
        type=str,
        default="How many babies named Anthony were born in Texas in 1990?",
        help="Natural language question"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable schema caching"
    )
    parser.add_argument(
        "--llm",
        type=str,
        choices=["llama", "qwen"],
        default="llama",
        help="LLM model to use"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
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
                    model_name=args.llm
                )
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
    else:
        run_demo(
            question=args.question,
            use_cache=not args.no_cache,
            model_name=args.llm
        )


if __name__ == "__main__":
    main()
