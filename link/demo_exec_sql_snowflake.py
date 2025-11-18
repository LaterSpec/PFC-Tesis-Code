"""
Demo: End-to-End Pipeline - Question → Schema → SQL → Execution

This demo shows the complete LinkAlign text-to-SQL pipeline:
1. inspect_schema: Load schema metadata and select relevant columns
2. gen_sql: Generate SQL from natural language question
3. exec_sql: Execute SQL safely and return results

Usage:
    python link/demo_exec_sql_snowflake.py --question "What are the top 5 most popular female names in California in 2000?"
    python link/demo_exec_sql_snowflake.py --question "How many babies were named John in 1950?" --llm qwen
    python link/demo_exec_sql_snowflake.py --interactive
"""
import argparse
import json
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from modules.inspect_schema import inspect_schema
from modules.gen_sql import gen_sql
from modules.exec_sql import exec_sql


def load_llm_pipeline(model_name: str = "llama"):
    """Load LLM pipeline and tokenizer."""
    from transformers import AutoTokenizer, pipeline
    
    if model_name == "llama":
        model_id = "meta-llama/Llama-3.2-3B-Instruct"
        print(f"[LLM] Loading {model_id}...")
    elif model_name == "qwen":
        model_id = "cycloneboy/CscSQL-Merge-Qwen2.5-Coder-1.5B-Instruct"
        print(f"[LLM] Loading {model_id}...")
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Load tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Load pipeline
    llm_pipeline = pipeline(
        "text-generation",
        model=model_id,
        device_map="auto",
        torch_dtype="auto"
    )
    
    print(f"[LLM] Model loaded successfully")
    return llm_pipeline, tokenizer


def print_schema_info(result):
    """Pretty print schema information."""
    print("\n" + "=" * 80)
    print("SCHEMA CONTEXT")
    print("=" * 80)
    
    schema_context = result["schema_context"]
    selected = schema_context["selected_schema"]
    
    print(f"\nDatabase: {selected.db_id}")
    print(f"Tables: {len(selected.tables)}")
    print(f"Total columns: {sum(len(t.columns) for t in selected.tables)}")
    
    for table in selected.tables:
        print(f"\n  {table.table_name} ({len(table.columns)} columns):")
        for col in table.columns:
            # FIX: Use .name instead of .column_name
            needed_marker = "✓" if col.extra.get("needed", False) else " "
            print(f"    {needed_marker} {col.name} ({col.type})")
    
    # Show signals - FIX: Handle both dict and dataclass
    signals = result["schema_signals"]
    print("\n--- Schema Signals ---")
    
    # Safely extract values
    if isinstance(signals, dict):
        signal_dict = signals
    elif hasattr(signals, 'values'):
        signal_dict = signals.values
    else:
        signal_dict = {}
    
    for key, value in sorted(signal_dict.items()):
        if value > 0:  # Only show non-zero signals
            print(f"  {key}: {value:.3f}")


def print_generation_info(gen_result):
    """Pretty print SQL generation information."""
    print("\n" + "=" * 80)
    print("SQL GENERATION")
    print("=" * 80)
    
    print(f"\nGenerated SQL:")
    print(f"```sql")
    print(gen_result["sql"])
    print(f"```")
    
    print(f"\nExpected Shape:")
    expected_shape = gen_result.get("expected_shape", {})
    print(f"  Kind: {expected_shape.get('kind', 'unknown')}")
    print(f"  Rows: {expected_shape.get('rows', 'unknown')}")
    
    # Show signals - FIX: Proper extraction
    signals = gen_result.get("gen_signals", {})
    print("\n--- Generation Signals ---")
    
    if isinstance(signals, dict):
        signal_dict = signals
    elif hasattr(signals, 'values'):
        signal_dict = signals.values
    else:
        signal_dict = {}
    
    for key, value in sorted(signal_dict.items()):
        if value > 0:
            print(f"  {key}: {value:.3f}")


def print_execution_info(exec_result):
    """Pretty print execution results."""
    print("\n" + "=" * 80)
    print("EXECUTION RESULTS")
    print("=" * 80)
    
    exec_data = exec_result.get("execution", {})
    
    if exec_data.get("error"):
        print(f"\n[ERROR] {exec_data['error']}")
    else:
        print(f"\n✓ Query executed successfully")
        print(f"  Rows: {exec_data.get('row_count', 0)}")
        print(f"  Columns: {len(exec_data.get('columns', []))}")
        print(f"  Latency: {exec_data.get('latency_ms', 0):.2f}ms")
        
        # Show data
        rows = exec_data.get("rows", [])
        if rows:
            print(f"\n--- Results (showing {min(len(rows), 10)} of {exec_data.get('row_count', 0)} rows) ---")
            
            # Print as table
            columns = exec_data.get("columns", [])
            
            if columns:
                # Header
                header = " | ".join(f"{col:20}" for col in columns)
                print(f"  {header}")
                print(f"  {'-' * len(header)}")
                
                # Rows
                for row in rows[:10]:
                    row_str = " | ".join(f"{str(row.get(col, 'NULL')):20}" for col in columns)
                    print(f"  {row_str}")
        else:
            print("\n[WARN] No rows returned")
    
    # Validation info
    validation = exec_result.get("validation", {})
    if validation.get("errors"):
        print("\n--- Validation Errors ---")
        for error in validation["errors"]:
            print(f"  ✗ {error}")
    
    if validation.get("warnings"):
        print("\n--- Validation Warnings ---")
        for warning in validation["warnings"]:
            print(f"  ⚠ {warning}")
    
    # Show signals - FIX
    signals = exec_result.get("exec_signals", {})
    print("\n--- Execution Signals ---")
    
    if isinstance(signals, dict):
        signal_dict = signals
    elif hasattr(signals, 'values'):
        signal_dict = signals.values
    else:
        signal_dict = {}
    
    for key, value in sorted(signal_dict.items()):
        if value > 0:
            print(f"  {key}: {value:.3f}")


def print_combined_signals(schema_signals, gen_signals, exec_signals):
    """Print combined signals from all pipeline stages."""
    print("\n" + "=" * 80)
    print("COMBINED PIPELINE SIGNALS (for SCM)")
    print("=" * 80)
    
    all_signals = {}
    
    # Helper function to extract signal dict
    def extract_signals(sig, prefix):
        if isinstance(sig, dict):
            return {f"{prefix}_{k}": v for k, v in sig.items()}
        elif hasattr(sig, 'values'):
            return {f"{prefix}_{k}": v for k, v in sig.values.items()}
        else:
            return {}
    
    # Extract from each stage
    all_signals.update(extract_signals(schema_signals, "schema"))
    all_signals.update(extract_signals(gen_signals, "gen"))
    all_signals.update(extract_signals(exec_signals, "exec"))
    
    # Show all non-zero signals
    non_zero = {k: v for k, v in all_signals.items() if v > 0}
    print(f"\nTotal signals: {len(non_zero)}/{len(all_signals)}")
    
    for key, value in sorted(non_zero.items()):
        print(f"  {key}: {value:.3f}")


def run_pipeline(question: str, llm_pipeline, llm_tokenizer, db_config: dict, exec_config: dict, use_cache: bool = True):
    """Run complete pipeline: inspect → gen → exec."""
    
    print("\n" + "=" * 80)
    print(f"QUESTION: {question}")
    print("=" * 80)
    
    # === STAGE 1: INSPECT SCHEMA ===
    print("\n[STAGE 1/3] Schema Inspection...")
    
    try:
        # Configure inspect_schema with LLM
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
            "llm_tokenizer": llm_tokenizer
        }
        
        # Add use_cache to db_config
        db_config_with_cache = db_config.copy()
        db_config_with_cache["use_cache"] = use_cache
        
        inspect_result = inspect_schema(
            question=question,
            db_config=db_config_with_cache,
            inspect_config=inspect_config
        )
        
        print_schema_info(inspect_result)
        
        # Extract schema_context for next stages
        schema_context = inspect_result["schema_context"]
        
    except Exception as e:
        print(f"\n[ERROR] Stage 1 (inspect_schema) failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None
    
    # === STAGE 2: GENERATE SQL ===
    print("\n[STAGE 2/3] SQL Generation...")
    
    try:
        # Configure gen_sql with LLM
        gen_config = {
            "llm_pipeline": llm_pipeline,
            "llm_tokenizer": llm_tokenizer,
            "generation": {
                "max_new_tokens": 768,
                "temperature": 0.0,
                "do_sample": False,
                "repetition_penalty": 1.0,
            },
            "num_candidates": 1
        }
        
        gen_result = gen_sql(
            question=question,
            schema_context=schema_context,
            gen_config=gen_config
        )
        
        print_generation_info(gen_result)
        
    except Exception as e:
        print(f"\n[ERROR] Stage 2 (gen_sql) failed: {e}")
        import traceback
        traceback.print_exc()
        return inspect_result, None, None
    
    # === STAGE 3: EXECUTE SQL ===
    print("\n[STAGE 3/3] SQL Execution...")
    
    try:
        # Create exec-specific db_config with credential_path
        exec_db_config = {
            "engine": "snowflake",
            "credential_path": db_config.get("credential_path"),
            "database": db_config.get("databases", ["USA_NAMES"])[0],
            "schema": db_config.get("schemas", ["USA_NAMES"])[0]
        }
        
        exec_result = exec_sql(
            sql=gen_result["sql"],
            schema_context=schema_context,
            expected_shape=gen_result["expected_shape"],
            db_config=exec_db_config,
            exec_config=exec_config
        )
        
        print_execution_info(exec_result)
        
        # === COMBINED SIGNALS ===
        print_combined_signals(
            inspect_result["schema_signals"],
            gen_result["gen_signals"],
            exec_result["exec_signals"]
        )
        
        return inspect_result, gen_result, exec_result
        
    except Exception as e:
        print(f"\n[ERROR] Stage 3 (exec_sql) failed: {e}")
        import traceback
        traceback.print_exc()
        return inspect_result, gen_result, None


def main():
    parser = argparse.ArgumentParser(description="Demo: Complete text-to-SQL pipeline with execution")
    parser.add_argument("--question", type=str, help="Natural language question")
    parser.add_argument("--llm", choices=["llama", "qwen"], default="llama", help="LLM model to use")
    parser.add_argument("--no-cache", action="store_true", help="Disable schema caching")
    parser.add_argument("--max-rows", type=int, default=100, help="Max rows to fetch")
    parser.add_argument("--timeout", type=int, default=30, help="Query timeout in seconds")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode (ask multiple questions)")
    
    args = parser.parse_args()
    
    # === LOAD LLM ===
    llm_pipeline, llm_tokenizer = load_llm_pipeline(args.llm)
    
    # === CONFIGURE DATABASE ===
    # Credentials are in src/cred/ (one level up from link/)
    cred_file = Path(__file__).parent.parent / "src" / "cred" / "snowflake_credential.json"
    
    db_config = {
        "engine": "snowflake",
        "credential_path": str(cred_file),
        "databases": ["USA_NAMES"],
        "schemas": ["USA_NAMES"],
        "cache_path": str(Path(__file__).parent / "cache" / "schema_usa_names_snowflake.json")
    }
    
    # === CONFIGURE EXECUTION ===
    exec_config = {
        "max_rows": args.max_rows,
        "timeout": args.timeout,
        "strict_validation": False
    }
    
    # === RUN PIPELINE ===
    if args.interactive:
        print("\n=== INTERACTIVE MODE ===")
        print("Enter questions (or 'quit' to exit):")
        
        while True:
            try:
                question = input("\nQuestion: ").strip()
                if question.lower() in ["quit", "exit", "q"]:
                    break
                
                if not question:
                    continue
                
                run_pipeline(
                    question=question,
                    llm_pipeline=llm_pipeline,
                    llm_tokenizer=llm_tokenizer,
                    db_config=db_config,
                    exec_config=exec_config,
                    use_cache=not args.no_cache
                )
                
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"\n[ERROR] {e}")
                import traceback
                traceback.print_exc()
    
    else:
        # Single question mode
        if not args.question:
            # Default question
            question = "What are the top 5 most popular female names in California in 2000?"
            print(f"[INFO] Using default question: {question}")
        else:
            question = args.question
        
        run_pipeline(
            question=question,
            llm_pipeline=llm_pipeline,
            llm_tokenizer=llm_tokenizer,
            db_config=db_config,
            exec_config=exec_config,
            use_cache=not args.no_cache
        )
    
    print("\n=== DEMO COMPLETE ===")


if __name__ == "__main__":
    main()
