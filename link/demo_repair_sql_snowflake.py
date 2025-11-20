"""
Demo: Complete Pipeline with Repair - Question → Schema → SQL → Execution → Repair

This demo shows the full LinkAlign text-to-SQL pipeline with repair capability:
1. inspect_schema: Load schema metadata and select relevant columns
2. gen_sql: Generate SQL from natural language question
3. exec_sql: Execute SQL safely and return results
4. repair_sql: Detect and fix semantic issues if needed

Usage:
    python link/demo_repair_sql_snowflake.py --question "How many babies named John in California in 2000?"
    python link/demo_repair_sql_snowflake.py --interactive
"""
import argparse
import json
import os
import sys
from pathlib import Path

# Add repository root (link/) so imports like `src.modules...` resolve
sys.path.insert(0, str(Path(__file__).parent))

from src.modules.inspect_schema import inspect_schema
from src.modules.gen_sql import gen_sql
from src.modules.exec_sql import exec_sql
from src.modules.repair_sql import repair_sql, should_trigger_repair
from src.modules.repair_sql.config import RepairConfig


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
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
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
    print("STAGE 1/4: SCHEMA INSPECTION")
    print("=" * 80)
    
    schema_context = result["schema_context"]
    selected = schema_context["selected_schema"]
    
    print(f"\nDatabase: {selected.db_id}")
    print(f"Tables: {len(selected.tables)}")
    print(f"Total columns: {sum(len(t.columns) for t in selected.tables)}")
    
    for table in selected.tables:
        print(f"\n  {table.table_name} ({len(table.columns)} columns):")
        for col in table.columns[:10]:  # Limit display
            needed_marker = "✓" if col.extra.get("needed", False) else " "
            print(f"    {needed_marker} {col.name} ({col.type})")
    
    signals = result["schema_signals"]
    signal_dict = signals.values if hasattr(signals, 'values') else signals
    
    print("\n--- Schema Signals ---")
    for key, value in sorted(signal_dict.items()):
        if value > 0:
            print(f"  {key}: {value:.3f}")


def print_generation_info(gen_result):
    """Pretty print SQL generation information."""
    print("\n" + "=" * 80)
    print("STAGE 2/4: SQL GENERATION")
    print("=" * 80)
    
    print(f"\nGenerated SQL:")
    print(f"```sql")
    print(gen_result["sql"])
    print(f"```")
    
    print(f"\nExpected Shape: {gen_result['expected_shape']}")
    
    signals = gen_result["gen_signals"]
    signal_dict = signals.values if hasattr(signals, 'values') else signals
    
    print("\n--- Generation Signals ---")
    for key, value in sorted(signal_dict.items()):
        if value > 0:
            print(f"  {key}: {value:.3f}")


def print_execution_info(exec_result):
    """Pretty print execution results."""
    print("\n" + "=" * 80)
    print("STAGE 3/4: SQL EXECUTION")
    print("=" * 80)
    
    result = exec_result["result"]
    
    if result.error:
        print(f"\n❌ Execution Error: {result.error}")
    else:
        print(f"\n✓ Execution successful!")
        print(f"  Rows returned: {result.row_count}")
        print(f"  Latency: {result.latency_ms:.2f} ms")
        print(f"  Columns: {', '.join(result.columns)}")
        
        if result.rows:
            print(f"\n--- Sample Results (first 5 rows) ---")
            for i, row in enumerate(result.rows[:5], 1):
                print(f"  Row {i}: {row}")
    
    signals = exec_result["exec_signals"]
    signal_dict = signals.values if hasattr(signals, 'values') else signals
    
    print("\n--- Execution Signals ---")
    for key, value in sorted(signal_dict.items()):
        if value != 0:
            print(f"  {key}: {value:.3f}")


def print_repair_info(repair_result):
    """Pretty print repair results."""
    print("\n" + "=" * 80)
    print("STAGE 4/4: SQL REPAIR")
    print("=" * 80)
    
    if not repair_result.applied:
        print("\n⊘ No repair applied")
        if repair_result.issues:
            print(f"  Issues detected: {len(repair_result.issues)}")
            for issue in repair_result.issues:
                print(f"    - {issue.issue_type}: {issue.column}")
        else:
            print("  No issues detected")
        return
    
    print(f"\n✓ Repair applied!")
    print(f"  Issues detected: {len(repair_result.issues)}")
    for issue in repair_result.issues:
        print(f"    - {issue.issue_type}: {issue.column} = {issue.value_used}")
    
    if repair_result.patch:
        print(f"\n  Patch: {repair_result.patch.description}")
        print(f"\n  Repaired SQL:")
        print(f"  ```sql")
        print(f"  {repair_result.repaired_sql}")
        print(f"  ```")
    
    if repair_result.repaired_exec_result:
        result = repair_result.repaired_exec_result
        print(f"\n--- Repaired Execution Results ---")
        print(f"  Rows returned: {result.row_count}")
        print(f"  Latency: {result.latency_ms:.2f} ms")
        
        if result.rows:
            print(f"\n  Sample Results (first 5 rows):")
            for i, row in enumerate(result.rows[:5], 1):
                print(f"    Row {i}: {row}")
    
    print("\n--- Repair Signals ---")
    for key, value in sorted(repair_result.repair_signals.items()):
        if value != 0:
            print(f"  {key}: {value:.3f}")


def run_pipeline(question: str, llm_pipeline, tokenizer, snowflake_config, repair_config):
    """Run the complete pipeline with repair."""
    
    # Stage 1: Inspect Schema
    print(f"\n🔍 Question: {question}")
    
    inspect_result = inspect_schema(
        question=question,
        db_config=snowflake_config,
        llm_pipeline=llm_pipeline,
        tokenizer=tokenizer,
    )
    
    print_schema_info(inspect_result)
    
    # Stage 2: Generate SQL
    gen_result = gen_sql(
        question=question,
        schema_context=inspect_result["schema_context"],
        llm_pipeline=llm_pipeline,
        tokenizer=tokenizer,
        engine="snowflake",
    )
    
    print_generation_info(gen_result)
    
    # Stage 3: Execute SQL
    exec_result = exec_sql(
        question=question,
        sql=gen_result["sql"],
        expected_shape=gen_result["expected_shape"],
        db_config=snowflake_config,
        engine="snowflake",
    )
    
    print_execution_info(exec_result)
    
    # Stage 4: Repair SQL (if needed)
    gen_signals = gen_result["gen_signals"]
    exec_signals = exec_result["exec_signals"]
    
    gen_signal_dict = gen_signals.values if hasattr(gen_signals, 'values') else gen_signals
    exec_signal_dict = exec_signals.values if hasattr(exec_signals, 'values') else exec_signals
    
    if should_trigger_repair(gen_signal_dict, exec_signal_dict):
        print("\n🔧 Triggering repair module...")
        
        # Create exec_runner for repair
        def exec_runner(sql: str):
            # Use runner.run_sql_query to get ExecutionResult dataclass directly
            from src.modules.exec_sql.runner import run_sql_query
            return run_sql_query(
                sql=sql,
                schema_context=inspect_result["schema_context"],
                db_config=snowflake_config,
                exec_config={"max_rows": 100, "timeout": 30},
            )
        
        repair_config.exec_runner = exec_runner
        
        repair_result = repair_sql(
            question=question,
            original_sql=gen_result["sql"],
            expected_shape=gen_result["expected_shape"],
            schema_context=inspect_result["schema_context"],
            gen_signals=gen_signal_dict,
            exec_result=exec_result["result"],
            exec_signals=exec_signal_dict,
            engine="snowflake",
            db_config=snowflake_config,
            config=repair_config,
        )
        
        print_repair_info(repair_result)
    else:
        print("\n⊘ Repair not needed (signals indicate acceptable result)")
    
    print("\n" + "=" * 80)
    print("Pipeline complete!")
    print("=" * 80)


def interactive_mode(llm_pipeline, tokenizer, snowflake_config, repair_config):
    """Run in interactive mode."""
    print("\n" + "=" * 80)
    print("Interactive Mode - Enter questions (or 'quit' to exit)")
    print("=" * 80)
    
    while True:
        try:
            question = input("\n📝 Question: ").strip()
            
            if question.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            
            if not question:
                continue
            
            run_pipeline(question, llm_pipeline, tokenizer, snowflake_config, repair_config)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Demo: Complete Pipeline with Repair")
    parser.add_argument("--question", type=str, help="Natural language question")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--llm", choices=["llama", "qwen"], default="llama", help="LLM model to use")
    parser.add_argument("--credential", default="src/cred/snowflake_credential.json", help="Snowflake credential file")
    parser.add_argument("--enable-llm-mapper", action="store_true", help="Enable LLM for enum mapping")
    
    args = parser.parse_args()
    
    # Load LLM
    llm_pipeline, tokenizer = load_llm_pipeline(args.llm)
    
    # Load Snowflake config
    credential_path = Path(__file__).parent / args.credential
    if not credential_path.exists():
        raise FileNotFoundError(f"Credential file not found: {credential_path}")
    
    with open(credential_path) as f:
        snowflake_config = json.load(f)
    
    snowflake_config["engine"] = "snowflake"
    
    # Configure repair module
    repair_config = RepairConfig(
        enable_enum_repairs=True,
        enable_year_repairs=True,
        enable_llm_enum_mapper=args.enable_llm_mapper,
        llm_mapper_pipeline=llm_pipeline if args.enable_llm_mapper else None,
        llm_mapper_tokenizer=tokenizer if args.enable_llm_mapper else None,
    )
    
    # Run pipeline
    if args.interactive:
        interactive_mode(llm_pipeline, tokenizer, snowflake_config, repair_config)
    elif args.question:
        run_pipeline(args.question, llm_pipeline, tokenizer, snowflake_config, repair_config)
    else:
        # Default example
        example_question = "How many babies named John were born in California in 2000?"
        print(f"\nRunning example question: {example_question}")
        run_pipeline(example_question, llm_pipeline, tokenizer, snowflake_config, repair_config)


if __name__ == "__main__":
    main()
