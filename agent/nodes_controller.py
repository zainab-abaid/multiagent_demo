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
                # Think steps are internal reasoning → auto-advance
                state["step_cursor"] = step_cursor + 1
            elif action_type not in ("think", "tool_call", "answer"):
                # Unknown action type → skip it to avoid being stuck
                state["step_cursor"] = step_cursor + 1
            # NOTE: we do NOT advance cursor on "answer" here
    
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
    # Hard stop if episode marked done
    if state.get("done", False):
        return "end"
    
    if not state.get("plan") or "steps" not in state.get("plan", {}):
        # No plan → just answer once and finish
        state["done"] = True
        return "answer"
    
    steps = state["plan"]["steps"]
    step_cursor = state.get("step_cursor", 0)
    
    # --- END OF PLAN HANDLING ---
    if step_cursor >= len(steps):
        # All steps consumed
        if state.get("is_ready_for_answer", False):
            # We are ready and nothing else to do → end episode
            state["done"] = True
            return "end"
        else:
            # Not ready → maybe replan, but with a cap
            replan_count = state.get("replan_count", 0)
            max_replans = state.get("config", {}).get("max_replans", 3)
            
            if replan_count >= max_replans:
                # Hit replan limit → force readiness and answer once
                state["is_ready_for_answer"] = True
                state["done"] = True
                return "answer"
            
            return "planner"
    
    # --- INSIDE PLAN ---
    current_step = steps[step_cursor]
    action_type = current_step.get("action_type")
    
    if action_type == "tool_call":
        return "tool_caller"
    
    elif action_type == "think":
        # Cursor will be advanced by plan_controller_node
        # If next is a tool_call → go run tool; else → go to answer
        if step_cursor + 1 < len(steps):
            next_step = steps[step_cursor + 1]
            if next_step.get("action_type") == "tool_call":
                return "tool_caller"
        # No following tool → we will answer once and end
        state["done"] = True
        return "answer"
    
    elif action_type == "answer":
        has_tool_results = (
            bool(state.get("tool_results"))
            or bool(state.get("sql_results"))
            or bool(state.get("rag_docs"))
            or bool(state.get("api_results"))
        )
        
        replan_count = state.get("replan_count", 0)
        max_replans = state.get("config", {}).get("max_replans", 3)
        
        # Case 1: tools ran and we might need replanning
        if has_tool_results and not state.get("is_ready_for_answer", False):
            if replan_count < max_replans:
                return "planner"
            else:
                # Hit replan limit → force readiness; answer once and end
                state["is_ready_for_answer"] = True
                state["done"] = True
                return "answer"
        
        # Case 2: no tools ever ran → nothing to replan, just allow answering once
        if not has_tool_results and not state.get("is_ready_for_answer", False):
            state["is_ready_for_answer"] = True
        
        # Either we were already ready_for_answer or we just set it above:
        # in both cases, answer ONCE and then end.
        state["done"] = True
        return "answer"
    
    else:
        # Unknown action type → treat as no-op and move on next tick
        # plan_controller_node will increment the cursor
        state["done"] = True
        return "answer"
