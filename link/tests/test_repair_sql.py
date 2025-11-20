"""Unit tests for the repair_sql module.

Run with: pytest link/tests/test_repair_sql.py -v
"""
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.modules.repair_sql import repair_sql, should_trigger_repair
from src.modules.repair_sql.config import RepairConfig
from src.modules.repair_sql.types import (
    ColumnProfile,
    RepairInput,
    RepairIssue,
    RepairPatch,
    RepairResult,
)
from src.modules.repair_sql import detection, mappers, rules
from src.modules.exec_sql.types import ExecutionResult
from src.modules.inspect_schema.types import (
    ColumnMetadata,
    DatabaseMetadata,
    SchemaContext,
    TableMetadata,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_schema_context():
    """Create a sample SchemaContext for testing."""
    columns = [
        ColumnMetadata(
            name="name",
            type="VARCHAR",
            description="Given name",
            extra={
                "sample_values": ["John", "Mary", "James"],
                "profile": ColumnProfile(
                    semantic_role="free_text",
                    safe_for_enum_constraints=False,
                ),
            },
        ),
        ColumnMetadata(
            name="state",
            type="VARCHAR",
            description="US state code",
            extra={
                "sample_values": ["CA", "TX", "NY", "FL"],
                "profile": ColumnProfile(
                    semantic_role="enum",
                    safe_for_enum_constraints=True,
                    safe_for_repair_mapping=True,
                ),
            },
        ),
        ColumnMetadata(
            name="gender",
            type="VARCHAR",
            description="M or F",
            extra={
                "sample_values": ["M", "F"],
                "profile": ColumnProfile(
                    semantic_role="enum",
                    safe_for_enum_constraints=True,
                    safe_for_repair_mapping=True,
                ),
            },
        ),
        ColumnMetadata(
            name="year",
            type="INT",
            description="Year of birth",
            extra={
                "profile": ColumnProfile(
                    semantic_role="temporal",
                ),
            },
        ),
        ColumnMetadata(
            name="number",
            type="INT",
            description="Count of babies",
        ),
    ]

    table = TableMetadata(
        table_name="usa_names",
        full_name="PUBLIC.usa_names",
        columns=columns,
        description="USA baby names",
    )

    db = DatabaseMetadata(
        db_id="names_db",
        engine="snowflake",
        tables=[table],
    )

    return SchemaContext(
        schema_text="CREATE TABLE usa_names...",
        selected_schema=db,
        tokens_estimate=100,
    )


@pytest.fixture
def sample_config():
    """Create a sample RepairConfig."""
    return RepairConfig(
        enable_enum_repairs=True,
        enable_year_repairs=True,
        enable_llm_enum_mapper=False,
        max_enum_values_per_column=100,
    )


# ============================================================================
# Test: detection.py
# ============================================================================

def test_detect_missing_year_filter(sample_schema_context, sample_config):
    """Test detection of missing year filter."""
    question = "How many babies named John were born in 2000?"
    original_sql = 'SELECT COUNT(*) FROM usa_names WHERE "name" = \'John\''

    repair_input = RepairInput(
        question=question,
        original_sql=original_sql,
        expected_shape={"kind": "scalar"},
        schema_context=sample_schema_context,
        gen_signals={},
        exec_result=ExecutionResult(
            sql=original_sql,
            row_count=0,
            rows=[],
            columns=[],
        ),
        exec_signals={},
        engine="snowflake",
        config=sample_config,
    )

    issues = detection.detect_issues(repair_input)

    # Should detect missing year filter
    year_issues = [i for i in issues if i.issue_type == "missing_year_filter"]
    assert len(year_issues) == 1
    assert year_issues[0].question_value == 2000
    assert year_issues[0].column == "year"


def test_detect_enum_value_mismatch(sample_schema_context, sample_config):
    """Test detection of invalid enum values."""
    question = "How many babies were born in California?"
    original_sql = 'SELECT COUNT(*) FROM usa_names WHERE "state" = \'California\''

    repair_input = RepairInput(
        question=question,
        original_sql=original_sql,
        expected_shape={"kind": "scalar"},
        schema_context=sample_schema_context,
        gen_signals={},
        exec_result=ExecutionResult(
            sql=original_sql,
            row_count=0,
            rows=[],
            columns=[],
        ),
        exec_signals={},
        engine="snowflake",
        config=sample_config,
    )

    issues = detection.detect_issues(repair_input)

    # Should detect enum mismatch
    enum_issues = [i for i in issues if i.issue_type == "enum_value_mismatch"]
    assert len(enum_issues) == 1
    assert enum_issues[0].column == "state"
    assert enum_issues[0].value_used == "California"
    assert "CA" in enum_issues[0].suggested_values


def test_no_issues_with_valid_sql(sample_schema_context, sample_config):
    """Test that valid SQL doesn't trigger issues."""
    question = "How many babies were born in CA in 2000?"
    original_sql = 'SELECT COUNT(*) FROM usa_names WHERE "state" = \'CA\' AND "year" = 2000'

    repair_input = RepairInput(
        question=question,
        original_sql=original_sql,
        expected_shape={"kind": "scalar"},
        schema_context=sample_schema_context,
        gen_signals={},
        exec_result=ExecutionResult(
            sql=original_sql,
            row_count=100,
            rows=[{"count": 100}],
            columns=["count"],
        ),
        exec_signals={},
        engine="snowflake",
        config=sample_config,
    )

    issues = detection.detect_issues(repair_input)

    # Should detect no issues
    assert len(issues) == 0


# ============================================================================
# Test: mappers.py
# ============================================================================

def test_map_state_with_dict():
    """Test dictionary-based state mapping."""
    column = ColumnMetadata(name="state", type="VARCHAR")

    # Test lowercase
    result = mappers.map_literal_to_enum_with_dict(column, "california")
    assert result is not None
    assert result.value == "CA"
    assert result.mechanism == "dict"

    # Test capitalized
    result = mappers.map_literal_to_enum_with_dict(column, "Texas")
    assert result is not None
    assert result.value == "TX"

    # Test invalid
    result = mappers.map_literal_to_enum_with_dict(column, "CA")
    assert result is None


def test_map_gender_with_dict():
    """Test dictionary-based gender mapping."""
    column = ColumnMetadata(name="gender", type="VARCHAR")

    result = mappers.map_literal_to_enum_with_dict(column, "female")
    assert result is not None
    assert result.value == "F"
    assert result.mechanism == "dict"

    result = mappers.map_literal_to_enum_with_dict(column, "male")
    assert result is not None
    assert result.value == "M"


def test_map_unknown_column():
    """Test mapping with unknown column returns None."""
    column = ColumnMetadata(name="unknown_col", type="VARCHAR")

    result = mappers.map_literal_to_enum_with_dict(column, "some_value")
    assert result is None


# ============================================================================
# Test: rules.py
# ============================================================================

def test_apply_year_repair(sample_schema_context, sample_config):
    """Test year repair rule."""
    question = "How many babies in 2000?"
    original_sql = 'SELECT COUNT(*) FROM usa_names'

    repair_input = RepairInput(
        question=question,
        original_sql=original_sql,
        expected_shape={"kind": "scalar"},
        schema_context=sample_schema_context,
        gen_signals={},
        exec_result=ExecutionResult(sql=original_sql),
        exec_signals={},
        engine="snowflake",
        config=sample_config,
    )

    # Create a year issue
    issue = RepairIssue(
        issue_type="missing_year_filter",
        column="year",
        question_value=2000,
    )

    patch = rules.apply_year_repair(repair_input, [issue])

    assert patch is not None
    assert '"year" = 2000' in patch.new_sql
    assert "WHERE" in patch.new_sql
    assert patch.metadata["rule"] == "year_filter"
    assert issue in patch.issues_resolved


def test_apply_enum_repairs(sample_schema_context, sample_config):
    """Test enum repair rule."""
    question = "How many babies in California?"
    original_sql = 'SELECT COUNT(*) FROM usa_names WHERE "state" = \'California\''

    repair_input = RepairInput(
        question=question,
        original_sql=original_sql,
        expected_shape={"kind": "scalar"},
        schema_context=sample_schema_context,
        gen_signals={},
        exec_result=ExecutionResult(sql=original_sql),
        exec_signals={},
        engine="snowflake",
        config=sample_config,
    )

    # Create enum mismatch issue
    issue = RepairIssue(
        issue_type="enum_value_mismatch",
        column="state",
        value_used="California",
        suggested_values=["CA", "TX", "NY"],
        details={
            "raw_literal": "'California'",
            "sample_values": ["CA", "TX", "NY"],
        },
    )

    patch = rules.apply_enum_repairs(repair_input, [issue])

    assert patch is not None
    assert "'CA'" in patch.new_sql
    assert "'California'" not in patch.new_sql
    assert patch.metadata["rule"] == "enum_mapping"
    assert "dict" in patch.metadata["enum_mapping_mechanisms"]


def test_apply_year_repair_with_existing_where(sample_schema_context, sample_config):
    """Test year repair when WHERE clause already exists."""
    question = "How many babies in Texas in 2000?"
    original_sql = 'SELECT COUNT(*) FROM usa_names WHERE "state" = \'TX\''

    repair_input = RepairInput(
        question=question,
        original_sql=original_sql,
        expected_shape={"kind": "scalar"},
        schema_context=sample_schema_context,
        gen_signals={},
        exec_result=ExecutionResult(sql=original_sql),
        exec_signals={},
        engine="snowflake",
        config=sample_config,
    )

    issue = RepairIssue(
        issue_type="missing_year_filter",
        column="year",
        question_value=2000,
    )

    patch = rules.apply_year_repair(repair_input, [issue])

    assert patch is not None
    assert ' AND "year" = 2000' in patch.new_sql
    assert '"state" = \'TX\'' in patch.new_sql


# ============================================================================
# Test: should_trigger_repair()
# ============================================================================

def test_should_trigger_repair_positive():
    """Test that repair is triggered for empty results with good format."""
    gen_signals = {
        "primary_format_ok": 1.0,
        "risk_score": 0.3,
    }
    exec_signals = {
        "rows_empty": 1.0,
        "exec_error": 0.0,
    }

    assert should_trigger_repair(gen_signals, exec_signals) is True


def test_should_trigger_repair_negative_exec_error():
    """Test that repair is not triggered if there was an execution error."""
    gen_signals = {
        "primary_format_ok": 1.0,
        "risk_score": 0.3,
    }
    exec_signals = {
        "rows_empty": 1.0,
        "exec_error": 1.0,  # Error!
    }

    assert should_trigger_repair(gen_signals, exec_signals) is False


def test_should_trigger_repair_negative_bad_format():
    """Test that repair is not triggered if SQL format is bad."""
    gen_signals = {
        "primary_format_ok": 0.0,  # Bad format!
        "risk_score": 0.3,
    }
    exec_signals = {
        "rows_empty": 1.0,
        "exec_error": 0.0,
    }

    assert should_trigger_repair(gen_signals, exec_signals) is False


def test_should_trigger_repair_negative_high_risk():
    """Test that repair is not triggered if risk score is too high."""
    gen_signals = {
        "primary_format_ok": 1.0,
        "risk_score": 0.8,  # High risk!
    }
    exec_signals = {
        "rows_empty": 1.0,
        "exec_error": 0.0,
    }

    assert should_trigger_repair(gen_signals, exec_signals) is False


def test_should_trigger_repair_negative_has_rows():
    """Test that repair is not triggered if query returned rows."""
    gen_signals = {
        "primary_format_ok": 1.0,
        "risk_score": 0.3,
    }
    exec_signals = {
        "rows_empty": 0.0,  # Has rows!
        "exec_error": 0.0,
    }

    assert should_trigger_repair(gen_signals, exec_signals) is False


# ============================================================================
# Test: repair_sql() (Integration)
# ============================================================================

def test_repair_sql_year_filter(sample_schema_context):
    """Test end-to-end year filter repair."""
    question = "How many babies named John in 2000?"
    original_sql = 'SELECT COUNT(*) FROM usa_names WHERE "name" = \'John\''

    # Mock exec_runner
    def mock_exec_runner(sql: str) -> ExecutionResult:
        if "year" in sql:
            return ExecutionResult(
                sql=sql,
                row_count=50,
                rows=[{"count": 50}],
                columns=["count"],
                latency_ms=10.0,
            )
        return ExecutionResult(
            sql=sql,
            row_count=0,
            rows=[],
            columns=["count"],
            latency_ms=5.0,
        )

    config = RepairConfig(
        enable_year_repairs=True,
        enable_enum_repairs=False,
        exec_runner=mock_exec_runner,
    )

    result = repair_sql(
        question=question,
        original_sql=original_sql,
        expected_shape={"kind": "scalar"},
        schema_context=sample_schema_context,
        gen_signals={"primary_format_ok": 1.0, "risk_score": 0.3},
        exec_result=ExecutionResult(sql=original_sql, row_count=0),
        exec_signals={"rows_empty": 1.0, "exec_error": 0.0},
        engine="snowflake",
        config=config,
    )

    assert result.applied is True
    assert result.repaired_sql is not None
    assert '"year" = 2000' in result.repaired_sql
    assert result.repaired_exec_result is not None
    assert result.repaired_exec_result.row_count == 50
    assert result.repair_signals["repair_applied"] == 1.0
    assert result.repair_signals["repair_success"] == 1.0
    assert result.repair_signals["repair_used_year_rule"] == 1.0


def test_repair_sql_enum_mapping(sample_schema_context):
    """Test end-to-end enum mapping repair."""
    question = "How many babies in California?"
    original_sql = 'SELECT COUNT(*) FROM usa_names WHERE "state" = \'California\''

    # Mock exec_runner
    def mock_exec_runner(sql: str) -> ExecutionResult:
        if "'CA'" in sql or '"CA"' in sql:
            return ExecutionResult(
                sql=sql,
                row_count=1000,
                rows=[{"count": 1000}],
                columns=["count"],
                latency_ms=15.0,
            )
        return ExecutionResult(
            sql=sql,
            row_count=0,
            rows=[],
            columns=["count"],
            latency_ms=5.0,
        )

    config = RepairConfig(
        enable_year_repairs=False,
        enable_enum_repairs=True,
        exec_runner=mock_exec_runner,
    )

    result = repair_sql(
        question=question,
        original_sql=original_sql,
        expected_shape={"kind": "scalar"},
        schema_context=sample_schema_context,
        gen_signals={"primary_format_ok": 1.0, "risk_score": 0.3},
        exec_result=ExecutionResult(sql=original_sql, row_count=0),
        exec_signals={"rows_empty": 1.0, "exec_error": 0.0},
        engine="snowflake",
        config=config,
    )

    assert result.applied is True
    assert result.repaired_sql is not None
    assert "'CA'" in result.repaired_sql
    assert "'California'" not in result.repaired_sql
    assert result.repaired_exec_result is not None
    assert result.repaired_exec_result.row_count == 1000
    assert result.repair_signals["repair_applied"] == 1.0
    assert result.repair_signals["repair_used_enum_rule"] == 1.0


def test_repair_sql_no_issues_detected(sample_schema_context):
    """Test repair_sql when no issues are detected."""
    question = "How many babies in CA?"
    original_sql = 'SELECT COUNT(*) FROM usa_names WHERE "state" = \'CA\''

    config = RepairConfig(
        enable_year_repairs=True,
        enable_enum_repairs=True,
    )

    result = repair_sql(
        question=question,
        original_sql=original_sql,
        expected_shape={"kind": "scalar"},
        schema_context=sample_schema_context,
        gen_signals={"primary_format_ok": 1.0},
        exec_result=ExecutionResult(sql=original_sql, row_count=0),
        exec_signals={"rows_empty": 1.0},
        engine="snowflake",
        config=config,
    )

    assert result.applied is False
    assert result.repaired_sql is None
    assert result.repaired_exec_result is None
    assert result.repair_signals["repair_applied"] == 0.0


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
