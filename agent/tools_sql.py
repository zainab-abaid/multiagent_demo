"""SQL tool wrapper for Chinook database queries."""

import pathlib
import logging
import requests
from typing import Optional

from langchain.chat_models import init_chat_model
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit

logger = logging.getLogger(__name__)

CHINOOK_DB_URL = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"
CHINOOK_DB_PATH = "Chinook.db"

# Global DB instance (initialized on first use)
_db: Optional[SQLDatabase] = None
_run_query_tool = None
_schema_cache: Optional[str] = None


def _ensure_chinook_db(path: str = CHINOOK_DB_PATH) -> str:
    """Ensure Chinook.db exists locally; download it if needed."""
    db_path = pathlib.Path(path)
    if db_path.exists():
        logger.info(f"[DB] Using existing {db_path}")
        return str(db_path)

    logger.info(f"[DB] Downloading Chinook.db from {CHINOOK_DB_URL} ...")
    resp = requests.get(CHINOOK_DB_URL)
    resp.raise_for_status()
    db_path.write_bytes(resp.content)
    logger.info(f"[DB] Saved to {db_path}")
    return str(db_path)


def _init_db():
    """Initialize the database connection and tools."""
    global _db, _run_query_tool, _schema_cache
    
    if _db is not None:
        return _db, _run_query_tool
    
    db_path = _ensure_chinook_db()
    _db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
    logger.info(f"[DB] Dialect: {_db.dialect}")
    logger.info(f"[DB] Tables: {sorted(_db.get_usable_table_names())}")
    
    # Cache the schema for reuse
    _schema_cache = _db.get_table_info_no_throw()
    
    # Initialize LLM for toolkit (minimal, just for tool setup)
    # Use SQL_MODEL env var or default
    import os
    sql_model = os.getenv("SQL_MODEL", "gpt-4o-mini")
    llm = init_chat_model(sql_model)
    toolkit = SQLDatabaseToolkit(db=_db, llm=llm)
    tools = toolkit.get_tools()
    _run_query_tool = next(t for t in tools if t.name == "sql_db_query")
    
    return _db, _run_query_tool


def get_db_schema() -> str:
    """
    Get the database schema information.
    
    This can be called separately if the planner wants to fetch schema first,
    or it's automatically included when generating SQL queries.
    
    Returns
    -------
    str
        Database schema information including tables, columns, and relationships
    """
    db, _ = _init_db()
    global _schema_cache
    
    # Return cached schema if available, otherwise fetch fresh
    if _schema_cache is None:
        _schema_cache = db.get_table_info_no_throw()
    
    return _schema_cache


def sql_tool(query: str) -> dict:
    """
    Run a SQL query against the Chinook DB and return results.
    
    Parameters
    ----------
    query : str
        SQL query string to execute
        
    Returns
    -------
    dict
        Results in format: {"rows": [...], "columns": [...], "row_count": int}
    """
    db, run_query_tool = _init_db()
    
    try:
        # Execute the query using the tool
        tool_call = {
            "name": run_query_tool.name,
            "args": {"query": query},
            "id": "sql_query_call",
            "type": "tool_call",
        }
        result = run_query_tool.invoke(tool_call)
        
        # Parse the result
        # The tool returns a message-like object with content
        content = result.content if hasattr(result, 'content') else str(result)
        
        # Try to extract numeric values from the result for easier access
        import re
        numeric_values = []
        # Look for numbers (including decimals) in the result
        numbers = re.findall(r'\d+\.?\d*', content)
        if numbers:
            numeric_values = [float(n) for n in numbers if '.' in n or len(n) > 0]
        
        # Return structured result
        return {
            "query": query,
            "result": content,
            "success": True,
            "numeric_values": numeric_values,  # Extracted numbers for easier access
        }
        
    except Exception as e:
        logger.error(f"SQL query failed: {e}")
        return {
            "query": query,
            "result": None,
            "success": False,
            "error": str(e),
        }


def sql_tool_nl_to_sql(natural_language_query: str, model_name: str | None = None) -> dict:
    """
    Convert natural language to SQL and execute it.
    
    This is a convenience function that combines NL->SQL generation with execution.
    For now, we'll use a simple LLM call to generate SQL, then execute it.
    
    Parameters
    ----------
    natural_language_query : str
        Natural language question
    model_name : str
        LLM model to use for SQL generation
        
    Returns
    -------
    dict
        Results with both the generated SQL and execution results
    """
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain.chat_models import init_chat_model
    import os
    
    # Get model name from env if not provided, fallback to default
    if model_name is None:
        model_name = os.getenv("SQL_MODEL", "gpt-4o-mini")
    
    db, run_query_tool = _init_db()
    llm = init_chat_model(model_name)
    
    # Get schema for context (uses cached schema if available)
    schema = get_db_schema()
    
    system_prompt = f"""You are an assistant that answers questions by writing SQL queries against
the Chinook music store database.

Database schema:
{schema}

When the user asks a question, you should:
1. Write a syntactically correct SQL query.
2. Only select the columns needed to answer the question.
3. Avoid modifying the database (no INSERT, UPDATE, DELETE, DROP, etc.).

IMPORTANT RULES:
- Return ONLY ONE SQL query statement (SQLite can only execute one statement at a time).
- Only query data that exists in the Chinook database (artists, albums, tracks, invoices, customers, employees, etc.).
- Do NOT try to answer questions about company information, policies, or other non-database information in SQL - those should be handled by other tools.
- Do NOT perform currency conversions or multiply by conversion rates in SQL.
- Return totals in USD only (the database stores amounts in USD).
- Do NOT create calculated columns for currency conversions (e.g., do NOT multiply by 0.85, 1.18, etc.).
- Currency conversions should be handled by separate API tools, not in SQL queries.

If the user's question asks about multiple things (e.g., "X and Y"), write a SQL query for the part that requires database data only. Other information (like company policies, founding dates, etc.) should not be in the SQL query.

Return ONLY the SQL query, nothing else."""
    
    try:
        # Generate SQL
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=natural_language_query)
        ]
        response = llm.invoke(messages)
        sql_query = response.content.strip()
        
        # Remove markdown code blocks if present
        if sql_query.startswith("```"):
            lines = sql_query.split("\n")
            sql_query = "\n".join(lines[1:-1]) if len(lines) > 2 else sql_query
            sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
        
        # Handle multiple SQL statements - SQLite can only execute one at a time
        # Split by semicolon and take the first non-empty statement
        sql_statements = [stmt.strip() for stmt in sql_query.split(";") if stmt.strip()]
        if len(sql_statements) > 1:
            logger.warning(f"[SQL] Multiple SQL statements detected ({len(sql_statements)}), using only the first one. "
                         f"Full query: {sql_query[:200]}...")
            sql_query = sql_statements[0]
        
        # Execute the SQL
        execution_result = sql_tool(sql_query)
        
        # Format the result more clearly
        result_summary = {
            "natural_language_query": natural_language_query,
            "generated_sql": sql_query,
            "execution_result": execution_result,
        }
        
        # Extract the actual query result for easier access
        if isinstance(execution_result, dict) and "result" in execution_result:
            result_summary["query_result"] = execution_result["result"]
            result_summary["success"] = execution_result.get("success", True)
            # Pass through numeric values for easier extraction
            if "numeric_values" in execution_result:
                result_summary["numeric_values"] = execution_result["numeric_values"]
            
            # Log warning if SQL appears to contain currency conversion
            if "generated_sql" in result_summary:
                sql_lower = result_summary["generated_sql"].lower()
                if "* 0.85" in sql_lower or "* 1.18" in sql_lower or "eur" in sql_lower or "gbp" in sql_lower:
                    logger.warning(f"[SQL] Generated SQL appears to contain currency conversion. "
                                 f"This should be handled by API tools, not SQL. SQL: {result_summary['generated_sql'][:100]}")
        
        return result_summary
        
    except Exception as e:
        logger.error(f"NL to SQL conversion failed: {e}")
        return {
            "natural_language_query": natural_language_query,
            "generated_sql": None,
            "execution_result": None,
            "error": str(e),
        }

