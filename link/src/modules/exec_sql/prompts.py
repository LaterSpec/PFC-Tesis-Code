"""Optional prompts for LLM-based result explanation and auditing."""
from __future__ import annotations

from typing import Any, Dict, List


def build_result_explainer_prompt(
    question: str,
    sql: str,
    rows: List[Dict[str, Any]],
    row_count: int
) -> str:
    """
    Build prompt for LLM to explain query results in natural language.
    
    This is optional future work for improved user experience.
    The LLM can translate raw results into human-readable summaries.
    
    Args:
        question: Original natural language question
        sql: Executed SQL query
        rows: Result rows (may be truncated)
        row_count: Total number of rows
        
    Returns:
        Prompt string for LLM
    """
    rows_preview = rows[:5]  # Show only first 5 rows
    
    prompt = f"""You are a helpful assistant explaining database query results.

**User Question:** {question}

**SQL Query Executed:**
```sql
{sql}
```

**Results Summary:**
- Total rows: {row_count}
- Sample data: {rows_preview}

Please provide a clear, concise natural language explanation of what these results show.
Focus on answering the user's original question.
"""
    return prompt


def build_result_auditor_prompt(
    question: str,
    sql: str,
    rows: List[Dict[str, Any]],
    row_count: int
) -> str:
    """
    Build prompt for LLM to audit whether results actually answer the question.
    
    This is optional quality assurance for SCM.
    The LLM can detect cases where SQL executed successfully but didn't
    actually answer what was asked.
    
    Args:
        question: Original natural language question
        sql: Executed SQL query
        rows: Result rows (may be truncated)
        row_count: Total number of rows
        
    Returns:
        Prompt string for LLM requesting JSON response
    """
    rows_preview = rows[:10]  # Show first 10 rows for auditing
    
    prompt = f"""You are an expert database auditor. Your job is to verify if query results correctly answer the user's question.

**User Question:** {question}

**SQL Query:**
```sql
{sql}
```

**Results:**
- Row count: {row_count}
- Data sample: {rows_preview}

Analyze whether these results actually answer the question. Return JSON:
{{
    "answers_question": true/false,
    "coverage_score": 0.0-1.0,
    "explanation": "brief explanation of why/why not",
    "missing_aspects": ["list", "of", "missing", "information"]
}}

- answers_question: Does the result directly answer what was asked?
- coverage_score: How completely does it answer (0.0 = not at all, 1.0 = completely)
- explanation: Brief reasoning
- missing_aspects: What information is missing, if any
"""
    return prompt


def build_result_explainer_messages(
    question: str,
    sql: str,
    rows: List[Dict[str, Any]],
    row_count: int
) -> List[Dict[str, str]]:
    """
    Build chat-formatted messages for result explanation.
    
    Returns:
        List of message dicts for apply_chat_template
    """
    system_prompt = """You are a helpful assistant that explains database query results in clear, natural language. 
Focus on answering the user's original question based on the data returned."""
    
    user_prompt = f"""Original question: "{question}"

SQL executed:
```sql
{sql}
```

Results ({row_count} row(s)):
{rows[:5]}

Please explain what these results show."""
    
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]


def build_result_auditor_messages(
    question: str,
    sql: str,
    rows: List[Dict[str, Any]],
    row_count: int
) -> List[Dict[str, str]]:
    """
    Build chat-formatted messages for result auditing.
    
    Returns:
        List of message dicts for apply_chat_template
    """
    system_prompt = """You are an expert database auditor. Verify if query results correctly answer the user's question.
Return ONLY valid JSON with: {"answers_question": bool, "coverage_score": float, "explanation": str, "missing_aspects": list}"""
    
    user_prompt = f"""Question: "{question}"

SQL:
```sql
{sql}
```

Results ({row_count} row(s)):
{rows[:10]}

Audit whether this answers the question. Return JSON only."""
    
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
