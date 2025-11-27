# agent/api_router.py

import json
from typing import Any, Dict, List
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.chat_models import init_chat_model
from agent.tools_api_registry import API_TOOLS_REGISTRY

async def api_router_llm(user_query: str, state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Given the user query and current state (including previous tool results),
    decide which API tools to call and with what parameters.
    
    Returns a list of call specs:
    [
      {
        "tool": "<registry key>",
        "args": { ... }
      },
      ...
    ]
    """
    # Optionally give it a concise view of previous results.
    # Keep it very simple & textual to keep the prompt understandable.
    previous_summary = []
    
    # Check sql_results list first (contains ALL SQL queries that were executed)
    sql_results = state.get("sql_results", [])
    if sql_results:
        sql_summary_lines = []
        for idx, sql_res in enumerate(sql_results, 1):
            # Extract numeric values from this SQL query result
            numeric_vals = []
            
            # First, try to get numeric_values if stored directly
            stored_numeric_vals = sql_res.get("numeric_values", [])
            if stored_numeric_vals:
                numeric_vals.extend([float(v) for v in stored_numeric_vals if v is not None])
            
            # Also try to parse from query_result string if numeric_values not available
            if not numeric_vals:
                query_result = sql_res.get("query_result", "")
                if query_result:
                    # Try to parse numeric values from the query result string
                    # Format is typically like "[(412, 2328.6)]" or "[412]"
                    import re
                    numbers = re.findall(r'\d+\.?\d*', query_result)
                    for num_str in numbers:
                        try:
                            num_val = float(num_str)
                            if num_val not in numeric_vals:  # Avoid duplicates
                                numeric_vals.append(num_val)
                        except ValueError:
                            pass
            
            # Fallback to usd_value if nothing else found
            if not numeric_vals:
                usd_val = sql_res.get("usd_value")
                if usd_val is not None:
                    numeric_vals.append(usd_val)
            
            # Get the step query to understand context
            step_query = sql_res.get("step_query", "")
            generated_sql = sql_res.get("generated_sql", "")
            
            if numeric_vals:
                sql_summary_lines.append(
                    f"SQL_QUERY_{idx}: {step_query[:60]}... | "
                    f"SQL: {generated_sql[:80]}... | "
                    f"Values: {numeric_vals}"
                )
            else:
                sql_summary_lines.append(
                    f"SQL_QUERY_{idx}: {step_query[:60]}... | "
                    f"SQL: {generated_sql[:80]}... | "
                    f"No numeric values extracted"
                )
        
        if sql_summary_lines:
            previous_summary.append("SQL_TOOL RESULTS:")
            previous_summary.extend(sql_summary_lines)
    
    # Add RAG tool results summary
    if state.get("rag_docs"):
        previous_summary.append("RAG_TOOL: pricing / policy docs retrieved.")
    
    tools_description = []
    for name, meta in API_TOOLS_REGISTRY.items():
        tools_description.append({
            "name": name,
            "description": meta["description"],
            "schema": meta["schema"],
        })
    
    system_prompt = """You are an API routing assistant.

Your job:
- Look at the user query.
- Look at the previous tool results summary.
- Decide which LOW-LEVEL API tools to call and with what parameters.

You have access to these tools (names + JSON schemas). You MUST choose from them.

You MUST NOT invent new tool names or parameters outside the given schemas.

Return ONLY JSON of the form:
{
  "calls": [
    {
      "tool": "<tool name from the list>",
      "args": { ... }
    }
  ]
}

Rules:
- Use information from previous tool results (like numeric values) when setting parameters (e.g. amount_usd).
- If the query asks for a currency conversion, choose the appropriate currency conversion tool and set currencies correctly.
- If multiple API calls are needed, include multiple entries in "calls".
- IMPORTANT: If you see multiple SQL query results (e.g., SQL_QUERY_1, SQL_QUERY_2), you may need to COMBINE the values (e.g., multiply count * average, or sum multiple values) before using them in the API call.
- For example, if SQL_QUERY_1 returns [412] (count) and SQL_QUERY_2 returns [5.65] (average), and the user asks for total revenue, you should calculate: 412 * 5.65 = 2327.8, then use 2327.8 as the amount_usd parameter.
- Always calculate combined values when needed based on the user query context before making the API call.
"""
    
    user_prompt = f"""User query:
{user_query}

Previous tool results summary:
{chr(10).join(previous_summary) if previous_summary else "None"}

Available API tools:
{json.dumps(tools_description, indent=2)}

Decide which tools to call and with what arguments.

Return ONLY JSON, no markdown, no comments."""
    
    model_name = state.get("config", {}).get("api_router_model") or "gpt-4o-mini"
    llm = init_chat_model(model_name)
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    
    response = await llm.ainvoke(messages)
    text = response.content.strip()
    
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1])
        text = text.replace("```json", "").replace("```", "").strip()
    
    try:
        data = json.loads(text)
        calls = data.get("calls", [])
        # basic validation
        valid_calls = []
        for c in calls:
            if isinstance(c, dict) and "tool" in c and "args" in c:
                valid_calls.append(c)
        return valid_calls
    except Exception:
        # Fall back: no API calls
        return []

