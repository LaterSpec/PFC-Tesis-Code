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


def print_schema_info(schema_context):
    """Pretty print schema information."""
    print("\n" + "=" * 80)
    print("SCHEMA CONTEXT")
    print("=" * 80)
    
    selected = schema_context["selected_schema"]
    print(f"\nDatabase: {selected.db_id}")
    print(f"Tables: {len(selected.tables)}")
    print(f"Total columns: {sum(len(t.columns) for t in selected.tables)}")
    
    for table in selected.tables:
        print(f"\n  {table.table_name} ({len(table.columns)} columns):")
        for col in table.columns:
            print(f"    - {col.column_name} ({col.data_type})")
    
    # Show signals
    signals = schema_context["schema_signals"]
    print("\n--- Schema Signals ---")
    for key, value in sorted(signals.values.items()):
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
    print(f"  Kind: {gen_result['expected_shape']['kind']}")
    print(f"  Rows: {gen_result['expected_shape']['rows']}")
    
    # Show signals
    signals = gen_result["gen_signals"]
    print("\n--- Generation Signals ---")
    for key, value in sorted(signals.values.items()):
        if value > 0:
            print(f"  {key}: {value:.3f}")


def print_execution_info(exec_result):
    """Pretty print execution results."""
    print("\n" + "=" * 80)
    print("EXECUTION RESULTS")
    print("=" * 80)
    
    exec_data = exec_result["execution"]
    
    if exec_data["error"]:
        print(f"\n[ERROR] {exec_data['error']}")
    else:
        print(f"\n✓ Query executed successfully")
        print(f"  Rows: {exec_data['row_count']}")
        print(f"  Columns: {len(exec_data['columns'])}")
        print(f"  Latency: {exec_data['latency_ms']:.2f}ms")
        
        # Show data
        rows = exec_data["rows"]
        if rows:
            print(f"\n--- Results (showing {min(len(rows), 10)} of {exec_data['row_count']} rows) ---")
            
            # Print as table
            columns = exec_data["columns"]
            
            # Header
            header = " | ".join(f"{col:12}" for col in columns)
            print(f"  {header}")
            print(f"  {'-' * len(header)}")
            
            # Rows
            for row in rows[:10]:
                row_str = " | ".join(f"{str(row.get(col, 'NULL')):12}" for col in columns)
                print(f"  {row_str}")
        else:
            print("\n[WARN] No rows returned")
    
    # Validation info
    validation = exec_result["validation"]
    if validation["errors"]:
        print("\n--- Validation Errors ---")
        for error in validation["errors"]:
            print(f"  ✗ {error}")
    
    if validation["warnings"]:
        print("\n--- Validation Warnings ---")
        for warning in validation["warnings"]:
            print(f"  ⚠ {warning}")
    
    # Show signals
    signals = exec_result["exec_signals"]
    print("\n--- Execution Signals ---")
    for key, value in sorted(signals.values.items()):
        if value > 0:
            print(f"  {key}: {value:.3f}")


def print_combined_signals(schema_signals, gen_signals, exec_signals):
    """Print combined signals from all pipeline stages."""
    print("\n" + "=" * 80)
    print("COMBINED PIPELINE SIGNALS (for SCM)")
    print("=" * 80)
    
    all_signals = {}
    
    # Prefix each signal source
    for key, value in schema_signals.values.items():
        all_signals[f"schema_{key}"] = value
    
    for key, value in gen_signals.values.items():
        all_signals[f"gen_{key}"] = value
    
    for key, value in exec_signals.values.items():
        all_signals[f"exec_{key}"] = value
    
    # Show all non-zero signals
    print(f"\nTotal signals: {len([v for v in all_signals.values() if v > 0])}/{len(all_signals)}")
    
    for key, value in sorted(all_signals.items()):
        if value > 0:
            print(f"  {key}: {value:.3f}")


def run_pipeline(question: str, llm_config: dict, db_config: dict, exec_config: dict, use_cache: bool = True):
    """Run complete pipeline: inspect → gen → exec."""
    
    print("\n" + "=" * 80)
    print(f"QUESTION: {question}")
    print("=" * 80)
    
    # === STAGE 1: INSPECT SCHEMA ===
    print("\n[STAGE 1/3] Schema Inspection...")
    
    schema_context = inspect_schema(
        db_config=db_config,
        llm_config=llm_config,
        question=question,
        use_cache=use_cache
    )
    
    print_schema_info(schema_context)
    
    # === STAGE 2: GENERATE SQL ===
    print("\n[STAGE 2/3] SQL Generation...")
    
    gen_result = gen_sql(
        question=question,
        schema_context=schema_context,
        llm_config=llm_config,
        gen_config={"num_candidates": 1}
    )
    
    print_generation_info(gen_result)
    
    # === STAGE 3: EXECUTE SQL ===
    print("\n[STAGE 3/3] SQL Execution...")
    
    exec_result = exec_sql(
        sql=gen_result["sql"],
        schema_context=schema_context,
        expected_shape=gen_result["expected_shape"],
        db_config=db_config,
        exec_config=exec_config
    )
    
    print_execution_info(exec_result)
    
    # === COMBINED SIGNALS ===
    print_combined_signals(
        schema_context["schema_signals"],
        gen_result["gen_signals"],
        exec_result["exec_signals"]
    )
    
    return schema_context, gen_result, exec_result


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
    
    llm_config = {
        "llm_pipeline": llm_pipeline,
        "llm_tokenizer": llm_tokenizer,
        "max_new_tokens": 768,  # Enough for full JSON response
        "temperature": 0.1
    }
    
    # === CONFIGURE DATABASE ===
    cred_file = Path(__file__).parent / "src" / "cred" / "snowflake_credential.json"
    
    db_config = {
        "engine": "snowflake",
        "database": "USA_NAMES",
        "schema": "USA_NAMES",
        "credential_file": str(cred_file)
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
                    llm_config=llm_config,
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
            llm_config=llm_config,
            db_config=db_config,
            exec_config=exec_config,
            use_cache=not args.no_cache
        )
    
    print("\n=== DEMO COMPLETE ===")


if __name__ == "__main__":
    main()
