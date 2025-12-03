"""Tool caller node: executes tools specified in plan steps."""

import logging
from agent.state import AgentState
from agent.tracing import traced_tool_call
from agent.tools_sql import sql_tool_nl_to_sql
from agent.tools_rag import rag_tool
from agent.api_router import api_router_llm
from agent.tools_api_registry import API_TOOLS_REGISTRY

logger = logging.getLogger(__name__)


async def tool_caller_node(state: AgentState) -> AgentState:
    """
    Executes the tool specified in the current plan step.
    
    Handles:
    - sql_tool: NL→SQL conversion and execution
    - rag_tool: Document retrieval
    - api_tool: API router selects and calls concrete API tools
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
    tool_result = None
    tool_input = None
    
    try:
        feedback = state.get("feedback")
        
        if tool_name == "sql_tool":
            step_query = step.get("query") or state["user_query"]
            tool_input = {
                "natural_language_query": step_query,
                "feedback": feedback
            }
            # TODO: Remove debug logging after fixing execution issue
            logger.info(f"[TOOL_CALLER] Executing SQL tool for step {step_id}: {step_query[:100]}")
            tool_result = await sql_tool_nl_to_sql(
                natural_language_query=step_query,
                state=state,
                feedback=feedback
            )
            
            # TODO: Remove debug logging after fixing execution issue
            logger.info(f"[TOOL_CALLER] SQL tool result: generated_sql={tool_result.get('generated_sql') is not None}, query_result={tool_result.get('query_result') is not None}, error={tool_result.get('error')}")
            
            # Ensure sql_results is a list (handle case where it might be None)
            if state.get("sql_results") is None:
                state["sql_results"] = []
            state["sql_results"].append({
                "generated_sql": tool_result.get("generated_sql"),
                "query_result": tool_result.get("query_result"),
                "step_query": step_query,
            })
            
            # Ensure tool_results is a list
            if state.get("tool_results") is None:
                state["tool_results"] = []
            state["tool_results"].append({
                "step_id": step_id,
                "tool_name": "sql_tool",
                "raw_input": tool_input,
                "raw_output": tool_result,
            })
            
        elif tool_name == "rag_tool":
            tool_input = {"query": state["user_query"]}
            tool_result = traced_tool_call(
                node_name="tool_caller",
                state=state,
                tool_name="rag_tool",
                tool_callable=rag_tool,
                tool_input=tool_input,
            )
            state["rag_docs"] = tool_result
            # Ensure tool_results is a list
            if state.get("tool_results") is None:
                state["tool_results"] = []
            state["tool_results"].append({
                "step_id": step_id,
                "tool_name": "rag_tool",
                "raw_input": tool_input,
                "raw_output": tool_result,
            })
            
        elif tool_name == "api_tool":
            api_calls = await api_router_llm(state["user_query"], state, feedback=feedback)
            api_call_results = []
            
            for call in api_calls:
                registry_name = call["tool"]
                args = call.get("args", {})
                meta = API_TOOLS_REGISTRY.get(registry_name)
                
                if not meta:
                    api_call_results.append({
                        "tool": registry_name,
                        "error": "Unknown API tool in registry"
                    })
                    continue
                
                result = traced_tool_call(
                    node_name="tool_caller",
                    state=state,
                    tool_name=registry_name,
                    tool_callable=meta["fn"],
                    tool_input=args,
                )
                
                api_call_results.append({
                    "tool": registry_name,
                    "input": args,
                    "output": result,
                })
                
                # Ensure tool_results is a list
                if state.get("tool_results") is None:
                    state["tool_results"] = []
                state["tool_results"].append({
                    "step_id": step_id,
                    "tool_name": registry_name,
                    "raw_input": args,
                    "raw_output": result,
                })
            
            # Ensure api_results is a list
            if state.get("api_results") is None:
                state["api_results"] = []
            state["api_results"].extend(api_call_results)
            tool_result = api_call_results
            
        else:
            tool_result = {"error": f"Unknown high-level tool: {tool_name}"}
            state.setdefault("tool_results", []).append({
                "step_id": step_id,
                "tool_name": tool_name,
                "raw_input": None,
                "raw_output": tool_result,
            })
            
    except Exception as e:
        import traceback
        # TODO: Remove debug logging after fixing execution issue
        error_msg = f"Tool execution failed: {str(e)}\n{traceback.format_exc()}"
        logger.error(f"[TOOL_CALLER] {error_msg}")
        tool_result = {"error": str(e), "traceback": traceback.format_exc()}
        # Ensure tool_results is a list
        if state.get("tool_results") is None:
            state["tool_results"] = []
        state["tool_results"].append({
            "step_id": step_id,
            "tool_name": tool_name or "unknown",
            "raw_input": tool_input,
            "raw_output": tool_result,
        })
    
    state["latest_result"] = tool_result
    state["step_cursor"] = cursor + 1
    return state
