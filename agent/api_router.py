import json
from typing import Any, Dict, List
from langchain_core.messages import SystemMessage, HumanMessage
from agent.llm_utils import init_llm
from agent.tools_api_registry import API_TOOLS_REGISTRY
from agent.tracing import traced_llm_call

async def api_router_llm(user_query: str, state: Dict[str, Any], feedback: str = None) -> List[Dict[str, Any]]:
    """
    Routes to API tools based on user query and previous tool results.
    
    LLM reads RAG content to extract conversion rates, uses SQL results for numeric values,
    and selects appropriate API tools with parameters.
    
    Returns list of tool call specs: [{"tool": "...", "args": {...}}, ...]
    """
    previous_summary = []
    
    if feedback:
        previous_summary.append(f"PREVIOUS FEEDBACK/CRITIQUE: {feedback}")
    
    sql_results = state.get("sql_results", [])
    if sql_results:
        for idx, sql_res in enumerate(sql_results, 1):
            step_query = sql_res.get("step_query", "")
            generated_sql = sql_res.get("generated_sql", "")
            query_result = sql_res.get("query_result", "")
            
            summary = f"SQL_QUERY_{idx}: Query: {step_query} | SQL: {generated_sql[:80]}... | "
            if query_result:
                summary += f"Result: {query_result[:100]}..."
            else:
                summary += "No result"
            previous_summary.append(summary)
    
    # Get RAG documents from rag_results
    rag_results = state.get("rag_results") or []
    for rag_res in rag_results:
        docs = rag_res.get("docs", [])
        for rag_doc in docs:
            content = rag_doc.get("content", "")
            if content:
                previous_summary.append(f"RAG_TOOL: {content}")
    
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
- Use numeric values from previous tool results when setting parameters.
- For currency conversions: Read RAG_TOOL content to find conversion rates. Extract the rate value and include it as the "rate" parameter.
- If RAG content shows "1 USD = 0.92 EUR", use rate: 0.92 for EUR conversions.
- If RAG content is unavailable or no rate found, set rate to null.
- If multiple SQL results exist, combine values as needed by the user query (add, multiply, etc.).
"""
    
    user_prompt = f"""User query:
{user_query}

Previous tool results:
{'\n'.join(previous_summary) if previous_summary else "None"}

Available API tools:
{json.dumps(tools_description, indent=2)}

Decide which tools to call and with what arguments.

Return ONLY JSON, no markdown, no comments."""
    
    model_name = state.get("config", {}).get("api_router_model") or "gpt-4o-mini"
    llm, actual_model_name = init_llm(model_name)
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    
    response = await traced_llm_call(
        node_name="api_router",
        state=state,
        llm_callable=llm,
        llm_input=messages,
        model_name=actual_model_name
    )
    text = response.content.strip()
    
    # Remove markdown code blocks if LLM wrapped JSON in ```json ... ```
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1])
        text = text.replace("```json", "").replace("```", "").strip()
    
    try:
        data = json.loads(text)
        calls = data.get("calls", [])
        valid_calls = []
        for c in calls:
            if isinstance(c, dict) and "tool" in c and "args" in c:
                valid_calls.append(c)
        return valid_calls
    except Exception:
        return []

