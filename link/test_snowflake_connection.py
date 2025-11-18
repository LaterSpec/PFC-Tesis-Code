"""
Simple test script to verify Snowflake connection and schema loading.

This script tests:
1. Snowflake connection with credentials from JSON
2. Basic database/schema listing
3. Schema metadata loading via inspect_schema module

Usage:
    python link/test_snowflake_connection.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import json
import snowflake.connector


def test_basic_connection():
    """Test basic Snowflake connection using credentials JSON."""
    print("=" * 70)
    print("  Test 1: Basic Snowflake Connection")
    print("=" * 70)
    
    try:
        # Load credentials
        with open("src/cred/snowflake_credential.json", "r") as f:
            cred = json.load(f)
        
        print(f"\n✓ Credentials loaded")
        print(f"  Account: {cred['account']}")
        print(f"  User: {cred['username']}")
        print(f"  Warehouse: {cred['warehouse']}")
        print(f"  Role: {cred['role']}")
        
        # Connect to Snowflake
        conn = snowflake.connector.connect(
            account=cred["account"],
            user=cred["username"],
            password=cred["password"],
            warehouse=cred["warehouse"],
            role=cred["role"],
        )
        
        print(f"\n✓ Connected to Snowflake successfully")
        
        # Test basic query
        cur = conn.cursor()
        
        # List databases
        print("\n--- Available Databases ---")
        cur.execute("SHOW DATABASES")
        databases = cur.fetchall()
        for db in databases[:10]:  # Show first 10
            print(f"  • {db[1]}")  # db[1] is the database name
        
        # Check USA_NAMES database
        print("\n--- USA_NAMES Database ---")
        cur.execute("USE DATABASE USA_NAMES")
        cur.execute("SHOW SCHEMAS")
        schemas = cur.fetchall()
        print("Schemas:")
        for schema in schemas:
            print(f"  • {schema[1]}")  # schema[1] is the schema name
        
        # Check USA_NAMES schema
        print("\n--- USA_NAMES.USA_NAMES Schema ---")
        cur.execute("USE SCHEMA USA_NAMES.USA_NAMES")
        cur.execute("SHOW TABLES")
        tables = cur.fetchall()
        print("Tables:")
        for table in tables:
            print(f"  • {table[1]}")  # table[1] is the table name
        
        # Sample query on first table if available
        if tables:
            table_name = tables[0][1]
            print(f"\n--- Sample from {table_name} ---")
            cur.execute(f"SELECT * FROM {table_name} LIMIT 5")
            rows = cur.fetchall()
            cols = [desc[0] for desc in cur.description]
            print(f"Columns: {', '.join(cols)}")
            print("Sample rows:")
            for row in rows:
                print(f"  {row}")
        
        cur.close()
        conn.close()
        
        print("\n✓ Test 1 PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ Test 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metadata_loader():
    """Test the metadata_loader module with Snowflake."""
    print("\n" + "=" * 70)
    print("  Test 2: Metadata Loader (inspect_schema)")
    print("=" * 70)
    
    try:
        from src.modules.inspect_schema.metadata_loader import load_db_metadata
        
        # Configure for Snowflake
        db_config = {
            "engine": "snowflake",
            "credential_path": "src/cred/snowflake_credential.json",
            "databases": ["USA_NAMES"],
            "schemas": ["USA_NAMES"],
            "use_cache": False,  # Don't use cache for testing
        }
        
        print("\n• Loading metadata from Snowflake...")
        metadata = load_db_metadata(db_config)
        
        print(f"\n✓ Metadata loaded successfully")
        print(f"  Found {len(metadata)} database(s)")
        
        for db in metadata:
            print(f"\n--- Database: {db.db_id} ---")
            print(f"  Engine: {db.engine}")
            print(f"  Tables: {len(db.tables)}")
            
            for table in db.tables:
                print(f"\n  • Table: {table.table_name}")
                print(f"    Full name: {table.full_name}")
                print(f"    Columns: {len(table.columns)}")
                
                print("    Column details:")
                for col in table.columns[:5]:  # Show first 5 columns
                    nullable = col.extra.get("nullable", "?")
                    print(f"      - {col.name}: {col.type} (nullable: {nullable})")
                
                if len(table.columns) > 5:
                    print(f"      ... and {len(table.columns) - 5} more columns")
        
        print("\n✓ Test 2 PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ Test 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_pipeline():
    """Test the full inspect_schema pipeline with Snowflake."""
    print("\n" + "=" * 70)
    print("  Test 3: Full Pipeline (inspect_schema)")
    print("=" * 70)
    
    try:
        from src.modules.inspect_schema import inspect_schema
        
        question = "How many babies named Anthony were born in Texas in 1990?"
        print(f"\nQuestion: {question}")
        
        db_config = {
            "engine": "snowflake",
            "credential_path": "src/cred/snowflake_credential.json",
            "databases": ["USA_NAMES"],
            "schemas": ["USA_NAMES"],
            "use_cache": False,
        }
        
        inspect_config = {
            "max_candidates": 5,
            "max_final_schemas": 1,
            "serialization": {
                "style": "compact",
                "include_types": True,
                "include_descriptions": True
            },
            "extraction": {
                "mode": "heuristic",  # Use heuristic for quick test
            }
        }
        
        print("\n• Running inspect_schema pipeline...")
        result = inspect_schema(question, db_config, inspect_config)
        
        schema_context = result["schema_context"]
        schema_signals = result["schema_signals"]
        selected_schema = schema_context["selected_schema"]
        
        print(f"\n✓ Pipeline completed successfully")
        print(f"\n--- Results ---")
        print(f"Database: {selected_schema.db_id}")
        print(f"Engine: {selected_schema.engine}")
        print(f"Tables: {len(selected_schema.tables)}")
        print(f"Total columns: {int(schema_signals['num_columns_total'])}")
        print(f"Selected columns: {int(schema_signals['num_columns_selected'])}")
        
        print(f"\n--- Schema Text ---")
        print(schema_context["schema_text"])
        
        print("\n✓ Test 3 PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ Test 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("  SNOWFLAKE CONNECTION TEST SUITE")
    print("=" * 70)
    
    results = []
    
    # Test 1: Basic connection
    results.append(("Basic Connection", test_basic_connection()))
    
    # Test 2: Metadata loader
    results.append(("Metadata Loader", test_metadata_loader()))
    
    # Test 3: Full pipeline
    results.append(("Full Pipeline", test_full_pipeline()))
    
    # Summary
    print("\n" + "=" * 70)
    print("  TEST SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:30s} {status}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
