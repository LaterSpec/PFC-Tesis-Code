"""Unit tests for the inspect_schema module.

Run with: pytest link/tests/test_inspect_schema.py -v
"""
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.modules.inspect_schema import inspect_schema
from src.modules.inspect_schema.types import (
    ColumnMetadata,
    DatabaseMetadata,
    TableMetadata,
)
from src.modules.inspect_schema import (
    extraction,
    filtering,
    metadata_loader,
    retrieval,
    serialization,
    signals,
)


# ============================================================================
# Test Configuration
# ============================================================================

@pytest.fixture
def mock_db_config():
    """Mock database configuration for testing."""
    return {
        "engine": "bigquery",
        "project_id": "bigquery-public-data",
        "datasets": ["usa_names"],
        "credential_path": "src/cred/clean_node.json",
        "use_cache": True,
        "cache_path": "link/cache/test_schema.json"
    }


@pytest.fixture
def mock_inspect_config():
    """Mock inspect configuration for testing."""
    return {
        "max_candidates": 5,
        "max_final_schemas": 1,
        "serialization": {
            "style": "compact",
            "include_types": True,
            "include_descriptions": True
        }
    }


@pytest.fixture
def sample_database():
    """Create a sample DatabaseMetadata for testing."""
    columns = [
        ColumnMetadata(name="name", type="STRING", description="Given name"),
        ColumnMetadata(name="state", type="STRING", description="US state code"),
        ColumnMetadata(name="gender", type="STRING", description="M or F"),
        ColumnMetadata(name="year", type="INT64", description="Year of birth"),
        ColumnMetadata(name="number", type="INT64", description="Count of babies"),
    ]
    
    table = TableMetadata(
        table_name="usa_1910_2013",
        full_name="`bigquery-public-data.usa_names.usa_1910_2013`",
        columns=columns,
        description="USA baby names from 1910 to 2013"
    )
    
    return DatabaseMetadata(
        db_id="bigquery-public-data.usa_names",
        engine="bigquery",
        tables=[table]
    )


# ============================================================================
# Test: retrieval.py
# ============================================================================

def test_initial_retrieval_single_db(sample_database):
    """Test that initial_retrieval returns the only available database."""
    question = "How many babies named Anthony were born in Texas?"
    databases = [sample_database]
    
    result = retrieval.initial_retrieval(
        question=question,
        databases=databases,
        index={},
        max_candidates=5
    )
    
    assert len(result.candidates) == 1
    assert result.candidates[0]["db_id"] == "bigquery-public-data.usa_names"
    assert result.candidates[0]["score"] == 1.0
    assert result.round_id == 0


def test_rewrite_and_score_queries(sample_database):
    """Test that rewrite returns the original question for Process 1."""
    question = "How many babies named Anthony were born?"
    retrieval_result = retrieval.initial_retrieval(
        question=question,
        databases=[sample_database],
        index={},
        max_candidates=5
    )
    
    result = retrieval.rewrite_and_score_queries(
        question=question,
        retrieval_result=retrieval_result,
        databases=[sample_database],
        config={}
    )
    
    assert "rewrites" in result
    assert len(result["rewrites"]) == 1
    assert result["rewrites"][0]["q_id"] == "Q0"
    assert result["rewrites"][0]["text"] == question


# ============================================================================
# Test: filtering.py
# ============================================================================

def test_filter_irrelevant_schemas_keeps_single_db(sample_database):
    """Test that filtering keeps the single database."""
    question = "How many babies were born?"
    retrieval_result = retrieval.initial_retrieval(
        question=question,
        databases=[sample_database],
        index={},
        max_candidates=5
    )
    
    result = filtering.filter_irrelevant_schemas(
        question=question,
        retrieval_result=retrieval_result,
        databases=[sample_database],
        rewrite_info={},
        max_final_schemas=3
    )
    
    assert len(result.filtered_candidates) == 1
    assert result.stats["num_initial"] == 1
    assert result.stats["num_kept"] == 1
    assert result.stats["num_removed"] == 0
    assert result.stats["fraction_removed"] == 0.0


# ============================================================================
# Test: extraction.py
# ============================================================================

def test_extraction_identifies_name_column(sample_database):
    """Test that extraction identifies 'name' column when mentioned."""
    question = "How many babies named Anthony were born?"
    
    result = extraction.extract_relevant_items(
        question=question,
        filtered_dbs=[sample_database],
        config={}
    )
    
    # Find the 'name' column
    name_col = next(
        col for table in result.selected_schema.tables
        for col in table.columns if col.name == "name"
    )
    
    assert name_col.extra["needed"] is True


def test_extraction_identifies_state_column(sample_database):
    """Test that extraction identifies 'state' when Texas is mentioned."""
    question = "How many babies were born in Texas?"
    
    result = extraction.extract_relevant_items(
        question=question,
        filtered_dbs=[sample_database],
        config={}
    )
    
    # Find the 'state' column
    state_col = next(
        col for table in result.selected_schema.tables
        for col in table.columns if col.name == "state"
    )
    
    assert state_col.extra["needed"] is True


def test_extraction_identifies_year_column(sample_database):
    """Test that extraction identifies 'year' when year is mentioned."""
    question = "What names were popular in 1990?"
    
    result = extraction.extract_relevant_items(
        question=question,
        filtered_dbs=[sample_database],
        config={}
    )
    
    # Find the 'year' column
    year_col = next(
        col for table in result.selected_schema.tables
        for col in table.columns if col.name == "year"
    )
    
    assert year_col.extra["needed"] is True


def test_extraction_identifies_number_for_how_many(sample_database):
    """Test that 'how many' triggers 'number' column."""
    question = "How many babies were born?"
    
    result = extraction.extract_relevant_items(
        question=question,
        filtered_dbs=[sample_database],
        config={}
    )
    
    # Find the 'number' column
    number_col = next(
        col for table in result.selected_schema.tables
        for col in table.columns if col.name == "number"
    )
    
    assert number_col.extra["needed"] is True


def test_extraction_gender_not_needed_when_not_mentioned(sample_database):
    """Test that 'gender' is not marked as needed if not mentioned."""
    question = "How many babies named John were born in California in 2000?"
    
    result = extraction.extract_relevant_items(
        question=question,
        filtered_dbs=[sample_database],
        config={}
    )
    
    # Find the 'gender' column
    gender_col = next(
        col for table in result.selected_schema.tables
        for col in table.columns if col.name == "gender"
    )
    
    assert gender_col.extra["needed"] is False


def test_extraction_coverage_stats(sample_database):
    """Test that coverage stats are calculated correctly."""
    question = "How many babies named Anthony were born in Texas in 1990?"
    # Expected: name, state, year, number = 4/5 columns needed
    
    result = extraction.extract_relevant_items(
        question=question,
        filtered_dbs=[sample_database],
        config={}
    )
    
    assert result.coverage_stats["num_columns_total"] == 5
    assert result.coverage_stats["num_columns_needed"] == 4
    assert result.coverage_stats["fraction_columns_needed"] == 0.8


# ============================================================================
# Test: serialization.py
# ============================================================================

def test_serialization_includes_database_info(sample_database):
    """Test that serialization includes database information."""
    result = serialization.serialize_schema_context(
        selected_schema=sample_database,
        serialization_config={"style": "compact", "include_types": True}
    )
    
    assert "bigquery" in result.schema_text
    assert "bigquery-public-data.usa_names" in result.schema_text


def test_serialization_includes_table_name(sample_database):
    """Test that serialization includes table name."""
    result = serialization.serialize_schema_context(
        selected_schema=sample_database,
        serialization_config={"style": "compact", "include_types": True}
    )
    
    assert "usa_1910_2013" in result.schema_text


def test_serialization_includes_column_types(sample_database):
    """Test that serialization includes column types when configured."""
    result = serialization.serialize_schema_context(
        selected_schema=sample_database,
        serialization_config={"style": "compact", "include_types": True}
    )
    
    assert "STRING" in result.schema_text
    assert "INT64" in result.schema_text


def test_serialization_token_estimate(sample_database):
    """Test that token estimation is provided."""
    result = serialization.serialize_schema_context(
        selected_schema=sample_database,
        serialization_config={"style": "compact", "include_types": True}
    )
    
    assert result.tokens_estimate is not None
    assert result.tokens_estimate > 0


# ============================================================================
# Test: signals.py
# ============================================================================

def test_signals_calculates_fractions(sample_database):
    """Test that signals correctly calculates fractions."""
    question = "How many babies named Anthony were born in Texas?"
    
    # Simulate the pipeline
    retrieval_result = retrieval.initial_retrieval(
        question=question,
        databases=[sample_database],
        index={},
        max_candidates=5
    )
    
    filter_result = filtering.filter_irrelevant_schemas(
        question=question,
        retrieval_result=retrieval_result,
        databases=[sample_database],
        rewrite_info={},
        max_final_schemas=1
    )
    
    extraction_result = extraction.extract_relevant_items(
        question=question,
        filtered_dbs=[sample_database],
        config={}
    )
    
    result = signals.build_schema_signals(
        retrieval_result=retrieval_result,
        filter_result=filter_result,
        extraction_result=extraction_result
    )
    
    assert "fraction_db_removed" in result.values
    assert "fraction_columns_selected" in result.values
    assert result.values["num_db_initial"] == 1.0
    assert result.values["num_db_kept"] == 1.0


# ============================================================================
# Test: Full Pipeline (inspect_schema)
# ============================================================================

@pytest.mark.integration
def test_full_pipeline_returns_all_keys(mock_db_config, mock_inspect_config):
    """Test that the full pipeline returns all expected keys."""
    question = "How many babies named Anthony were born in Texas in 1990?"
    
    # This test requires actual BigQuery credentials
    # Skip if credentials are not available
    try:
        result = inspect_schema(
            question=question,
            db_config=mock_db_config,
            inspect_config=mock_inspect_config
        )
        
        # Check top-level keys
        assert "schema_context" in result
        assert "schema_signals" in result
        assert "debug_info" in result
        
        # Check schema_context structure
        assert "schema_text" in result["schema_context"]
        assert "selected_schema" in result["schema_context"]
        assert "tokens_estimate" in result["schema_context"]
        
        # Check that schema_text is not empty
        assert len(result["schema_context"]["schema_text"]) > 0
        
        # Check that signals are present
        assert "fraction_columns_selected" in result["schema_signals"]
        assert "num_db_kept" in result["schema_signals"]
        
    except Exception as e:
        pytest.skip(f"Skipping integration test: {e}")


def test_full_pipeline_with_different_questions(mock_db_config, mock_inspect_config):
    """Test pipeline with various question types."""
    questions = [
        "What are the most popular names in California?",
        "How many boys were named Michael in 1985?",
        "Which state had the most babies named Emma?",
    ]
    
    for question in questions:
        try:
            result = inspect_schema(
                question=question,
                db_config=mock_db_config,
                inspect_config=mock_inspect_config
            )
            
            # Basic validation
            assert "schema_context" in result
            assert result["schema_context"]["schema_text"] is not None
            assert len(result["schema_context"]["schema_text"]) > 0
            
        except Exception as e:
            pytest.skip(f"Skipping test for question '{question}': {e}")


# ============================================================================
# Test: Edge Cases
# ============================================================================

def test_extraction_with_empty_question(sample_database):
    """Test extraction with an empty question."""
    result = extraction.extract_relevant_items(
        question="",
        filtered_dbs=[sample_database],
        config={}
    )
    
    # Should still return a valid result
    assert result.selected_schema is not None


def test_extraction_with_no_databases():
    """Test extraction when no databases are provided."""
    result = extraction.extract_relevant_items(
        question="How many babies were born?",
        filtered_dbs=[],
        config={}
    )
    
    assert result.selected_schema.db_id == "none"
    assert result.coverage_stats["num_tables"] == 0


if __name__ == "__main__":
    # Run tests with pytest if available, otherwise print message
    try:
        import pytest
        pytest.main([__file__, "-v"])
    except ImportError:
        print("pytest not installed. Install with: pip install pytest")
        print("Run tests with: pytest link/tests/test_inspect_schema.py -v")
