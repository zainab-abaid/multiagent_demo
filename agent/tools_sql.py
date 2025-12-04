"""SQL tool wrapper for Chinook database queries."""

import os
import pathlib
import logging
import requests
from typing import Optional
from langchain_core.messages import HumanMessage, SystemMessage

from agent.llm_utils import init_llm
from agent.state import AgentState
from agent.tracing import traced_llm_call
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit

logger = logging.getLogger(__name__)

CHINOOK_DB_URL = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"
CHINOOK_DB_PATH = "Chinook.db"

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
    
    _schema_cache = _db.get_table_info_no_throw()
    
    sql_model = os.getenv("SQL_MODEL", "gpt-4o-mini")
    llm, _ = init_llm(sql_model)
    toolkit = SQLDatabaseToolkit(db=_db, llm=llm)
    tools = toolkit.get_tools()
    _run_query_tool = next(t for t in tools if t.name == "sql_db_query")
    
    return _db, _run_query_tool


def get_db_schema() -> str:
    """Get the database schema information."""
    db, _ = _init_db()
    global _schema_cache
    
    if _schema_cache is None:
        _schema_cache = db.get_table_info_no_throw()
    
    return _schema_cache


def sql_tool(query: str) -> dict:
    """Run a SQL query against the Chinook DB and return results."""
    db, run_query_tool = _init_db()
    
    try:
        tool_call = {
            "name": run_query_tool.name,
            "args": {"query": query},
            "id": "sql_query_call",
            "type": "tool_call",
        }
        result = run_query_tool.invoke(tool_call)
        content = result.content if hasattr(result, 'content') else str(result)
        
        return {
            "query": query,
            "result": content,
            "success": True,
        }
        
    except Exception as e:
        logger.error(f"SQL query failed: {e}")
        return {
            "query": query,
            "result": None,
            "success": False,
            "error": str(e),
        }


async def sql_tool_nl_to_sql(natural_language_query: str, state: AgentState, model_name: str | None = None, feedback: str | None = None) -> dict:
    """Convert natural language to SQL and execute it."""
    
    if model_name is None:
        model_name = os.getenv("SQL_MODEL", "gpt-4o-mini")
    
    db, run_query_tool = _init_db()
    llm, actual_model_name = init_llm(model_name)
    schema = get_db_schema()
    
    feedback_section = ""
    if feedback:
        feedback_section = f"\nPREVIOUS MISTAKES/FEEDBACK:\n{feedback}\n\nUse this feedback to correct your previous SQL query."
    
    system_prompt = f"""You are an assistant that answers questions by writing SQL queries against
the Chinook music store database.

Database schema:
{schema}
{feedback_section}

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

CRITICAL: Handling comma-separated names in Composer/Artist fields:
- Comma-separated names in the database (e.g., "Steven Tyler, Richie Supa") represent COLLABORATIONS stored as single strings, not separate individuals.
- Use EXACT string matching (WHERE Composer = 'Name1, Name2') unless the query explicitly says "OR" or "either".
- Only use IN (...) or OR when the user explicitly asks for "either X or Y" or "X or Y".
- Examples:
  * "composed by Steven Tyler, Richie Supa" → WHERE Composer = 'Steven Tyler, Richie Supa' (exact match)
  * "composed by Steven Tyler or Richie Supa" → WHERE Composer IN ('Steven Tyler', 'Richie Supa') (either/or)
  * "composed by either Steven Tyler or Richie Supa" → WHERE Composer IN ('Steven Tyler', 'Richie Supa') (either/or)

If the user's question asks about multiple things (e.g., "X and Y"), write a SQL query for the part that requires database data only. Other information (like company policies, founding dates, etc.) should not be in the SQL query.

Return ONLY the SQL query, nothing else."""
    
    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=natural_language_query)
        ]
        
        response = await traced_llm_call(
            node_name="sql_tool_gen",
            state=state,
            llm_callable=llm,
            llm_input=messages,
            model_name=actual_model_name
        )
            
        sql_query = response.content.strip()
        
        # Remove markdown code blocks if LLM wrapped SQL in ```sql ... ```
        if sql_query.startswith("```"):
            lines = sql_query.split("\n")
            sql_query = "\n".join(lines[1:-1]) if len(lines) > 2 else sql_query
            sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
        
        # SQLite can only execute one statement at a time - take first if multiple
        sql_statements = [stmt.strip() for stmt in sql_query.split(";") if stmt.strip()]
        if len(sql_statements) > 1:
            logger.warning(f"[SQL] Multiple SQL statements detected ({len(sql_statements)}), using only the first one. "
                         f"Full query: {sql_query[:200]}...")
            sql_query = sql_statements[0]
        
        # Execute the generated SQL query
        # TODO: Remove debug logging after fixing execution issue
        logger.info(f"[SQL_TOOL] Executing SQL query: {sql_query[:200]}")
        execution_result = sql_tool(sql_query)
        logger.info(f"[SQL_TOOL] Execution result: success={execution_result.get('success')}, result={execution_result.get('result') is not None}, error={execution_result.get('error')}")
        
        # Build result summary with query, SQL, and execution results
        result_summary = {
            "natural_language_query": natural_language_query,
            "generated_sql": sql_query,
            "execution_result": execution_result,
        }
        
        # Extract query result from execution result for easier access
        if isinstance(execution_result, dict) and "result" in execution_result:
            result_summary["query_result"] = execution_result["result"]
            result_summary["success"] = execution_result.get("success", True)
        else:
            logger.warning(f"[SQL_TOOL] Execution result missing 'result' field: {execution_result}")
        
        return result_summary
        
    except Exception as e:
        logger.error(f"NL to SQL conversion failed: {e}")
        return {
            "natural_language_query": natural_language_query,
            "generated_sql": None,
            "execution_result": None,
            "error": str(e),
        }

