# agent/nodes_tools.py

from agent.state import AgentState
from agent.tracing import traced_tool_call
from agent.tools_sql import sql_tool_nl_to_sql
from agent.tools_rag import rag_tool
from agent.api_router import api_router_llm
from agent.tools_api_registry import API_TOOLS_REGISTRY

async def tool_caller_node(state: AgentState) -> AgentState:
    """
    Executes the tool specified in the current plan step.
    - For "sql_tool": NL→SQL agent + execution
    - For "rag_tool": RAG retriever
    - For "api_tool": calls an API router LLM which selects concrete API tools and args
    """
    if not state.get("plan") or "steps" not in state["plan"]:
        state["done"] = True
        return state
    
    steps = state["plan"]["steps"]
    cursor = state.get("step_cursor", 0)
    if cursor >= len(steps):
        state["done"] = True
        return state
    
    step = steps[cursor]
    if step.get("action_type") != "tool_call":
        state["step_cursor"] = cursor + 1
        return state
    
    tool_name = step.get("tool")
    step_id = step.get("id", cursor)
    tool_input = None
    tool_result = None
    
    try:
        feedback = state.get("feedback")
        
        if tool_name == "sql_tool":
            # Use step-specific query if provided, otherwise fall back to user_query
            step_query = step.get("query") or state["user_query"]
            # Create tool_input WITH state for the callable, but create clean copy for logging
            tool_input_with_state = {
                "natural_language_query": step_query, 
                "state": state,  # Pass state for tracing inside the tool
                "feedback": feedback
            }
            # Create clean copy without state for logging (to avoid circular reference)
            tool_input_clean = {
                "natural_language_query": step_query,
                "feedback": feedback
            }
            tool_result = traced_tool_call(
                node_name="tool_caller",
                state=state,
                tool_name="sql_tool",
                tool_callable=sql_tool_nl_to_sql,
                tool_input=tool_input_with_state,
            )
            # optional convenience view for downstream
            numeric_vals = tool_result.get("numeric_values")
            if not numeric_vals and isinstance(tool_result.get("execution_result"), dict):
                numeric_vals = tool_result["execution_result"].get("numeric_values")
            numeric_vals = numeric_vals or []
            usd_value = float(numeric_vals[0]) if numeric_vals else None
            
            # Store SQL result - accumulate multiple SQL queries
            sql_query_result = {
                "generated_sql": tool_result.get("generated_sql"),
                "query_result": tool_result.get("query_result"),
                "usd_value": usd_value,
                "numeric_values": numeric_vals,  # Store all numeric values, not just first one
                "step_query": step_query,
            }
            
            # Track all SQL queries - initialize list if needed
            if state.get("sql_results") is None:
                state["sql_results"] = []
            state["sql_results"].append(sql_query_result)
            
            # log entry - use clean copy without state to avoid circular reference
            state.setdefault("tool_results", []).append(
                {
                    "step_id": step_id,
                    "tool_name": "sql_tool",
                    "raw_input": tool_input_clean,
                    "raw_output": tool_result,
                }
            )
            
        elif tool_name == "rag_tool":
            tool_input = {
                "query": state["user_query"],
                "top_k": 5,
                "documents_path": "documents",
            }
            tool_result = traced_tool_call(
                node_name="tool_caller",
                state=state,
                tool_name="rag_tool",
                tool_callable=rag_tool,
                tool_input=tool_input,
            )
            state["rag_docs"] = tool_result
            state.setdefault("tool_results", []).append(
                {
                    "step_id": step_id,
                    "tool_name": "rag_tool",
                    "raw_input": tool_input,
                    "raw_output": tool_result,
                }
            )
            
        elif tool_name == "api_tool":
            # 1) Ask router LLM which concrete API tools to call
            api_calls = await api_router_llm(state["user_query"], state, feedback=feedback)
            api_call_results = []
            for call in api_calls:
                registry_name = call["tool"]
                args = call.get("args", {})
                meta = API_TOOLS_REGISTRY.get(registry_name)
                if not meta:
                    api_call_results.append(
                        {"tool": registry_name, "error": "Unknown API tool in registry"}
                    )
                    continue
                
                fn = meta["fn"]
                tool_input = args
                result = traced_tool_call(
                    node_name="tool_caller",
                    state=state,
                    tool_name=registry_name,
                    tool_callable=fn,
                    tool_input=tool_input,
                )
                api_call_results.append(
                    {
                        "tool": registry_name,
                        "input": args,
                        "output": result,
                    }
                )
                # Also log in generic tool_results for your judge / reflection
                state.setdefault("tool_results", []).append(
                    {
                        "step_id": step_id,
                        "tool_name": registry_name,
                        "raw_input": args,
                        "raw_output": result,
                    }
                )
            
            # Store all API results in a generic place
            if state.get("api_results") is None:
                state["api_results"] = []
            state["api_results"].extend(api_call_results)
            tool_result = api_call_results
            
        else:
            tool_result = {"error": f"Unknown high-level tool: {tool_name}"}
            state.setdefault("tool_results", []).append(
                {
                    "step_id": step_id,
                    "tool_name": tool_name,
                    "raw_input": None,
                    "raw_output": tool_result,
                }
            )
            
    except Exception as e:
        tool_result = {"error": str(e)}
        state.setdefault("tool_results", []).append(
            {
                "step_id": step_id,
                "tool_name": tool_name,
                "raw_input": tool_input,
                "raw_output": tool_result,
            }
        )
    
    state["latest_result"] = tool_result
    state["step_cursor"] = cursor + 1
    return state
