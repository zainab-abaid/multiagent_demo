"""Plan controller node for routing execution flow."""

from typing import Literal
from agent.state import AgentState


def plan_controller_node(state: AgentState) -> AgentState:
    """
    Plan controller node: examines plan and updates state if needed.
    
    This node just updates the state. The routing decision is made
    by the separate routing function.
    """
    if state.get("plan") and "steps" in state.get("plan", {}):
        steps = state["plan"]["steps"]
        step_cursor = state.get("step_cursor", 0)
        
        if step_cursor < len(steps):
            current_step = steps[step_cursor]
            action_type = current_step.get("action_type")
            
            if action_type == "think":
                state["step_cursor"] = step_cursor + 1
            elif action_type == "answer" and state.get("is_ready_for_answer", False):
                state["step_cursor"] = step_cursor + 1
    
    return state


def route_controller(state: AgentState) -> Literal["tool_caller", "planner", "answer", "end"]:
    """
    Routing function for plan_controller node.
    
    Returns:
    - "tool_caller" if next step is a tool_call
    - "planner" if all steps done but not ready for answer (replanning)
    - "answer" if next step is answer or ready for answer
    - "end" if done
    """
    if state.get("done", False):
        return "end"
    
    if not state.get("plan") or "steps" not in state.get("plan", {}):
        return "answer"
    
    steps = state["plan"]["steps"]
    step_cursor = state.get("step_cursor", 0)
    
    if step_cursor >= len(steps):
        if state.get("is_ready_for_answer", False):
            return "answer"
        else:
            replan_count = state.get("replan_count", 0)
            max_replans = state.get("config", {}).get("max_replans", 3)
            
            if replan_count >= max_replans:
                # Hit replan limit → force readiness so we don't loop
                state["is_ready_for_answer"] = True
                return "answer"
                
            return "planner"
    
    current_step = steps[step_cursor]
    action_type = current_step.get("action_type")
    
    if action_type == "tool_call":
        return "tool_caller"
    elif action_type == "think":
        if step_cursor + 1 < len(steps):
            next_step = steps[step_cursor + 1]
            if next_step.get("action_type") == "tool_call":
                return "tool_caller"
        return "answer"
    elif action_type == "answer":
        has_tool_results = bool(state.get("tool_results")) or bool(state.get("sql_results")) or bool(state.get("rag_docs")) or bool(state.get("api_results"))
        
        if has_tool_results and not state.get("is_ready_for_answer", False):
            replan_count = state.get("replan_count", 0)
            max_replans = state.get("config", {}).get("max_replans", 3)
            
            if replan_count < max_replans:
                return "planner"
            else:
                # Hit replan limit → force readiness so we don't loop on answer
                state["is_ready_for_answer"] = True
        
        return "answer"
    else:
        return "answer"

