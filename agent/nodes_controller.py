"""Plan controller node for routing execution flow."""

from typing import Literal
from agent.state import AgentState


def plan_controller_node(state: AgentState) -> AgentState:
    """
    Plan controller node: examines plan and updates state if needed.
    
    This node just updates the state. The routing decision is made
    by the separate routing function.
    """
    # For "think" steps, advance cursor
    if state.get("plan") and "steps" in state.get("plan", {}):
        steps = state["plan"]["steps"]
        step_cursor = state.get("step_cursor", 0)
        
        if step_cursor < len(steps):
            current_step = steps[step_cursor]
            action_type = current_step.get("action_type")
            
            # For "think" steps, advance cursor
            if action_type == "think":
                state["step_cursor"] = step_cursor + 1
    
    return state


def route_controller(state: AgentState) -> Literal["tool_caller", "answer", "end"]:
    """
    Routing function for plan_controller node.
    
    Returns:
    - "tool_caller" if next step is a tool_call
    - "answer" if next step is answer or no more steps
    - "end" if done
    """
    # Check if done
    if state.get("done", False):
        return "end"
    
    # Check if we have a plan
    if not state.get("plan") or "steps" not in state.get("plan", {}):
        # No plan, go to answer
        return "answer"
    
    steps = state["plan"]["steps"]
    step_cursor = state.get("step_cursor", 0)
    
    # Check if we've completed all steps
    if step_cursor >= len(steps):
        # All steps done, go to answer
        return "answer"
    
    # Get current step
    current_step = steps[step_cursor]
    action_type = current_step.get("action_type")
    
    # Route based on action type
    if action_type == "tool_call":
        return "tool_caller"
    elif action_type == "think":
        # After advancing cursor in the node, check next step
        if step_cursor + 1 < len(steps):
            next_step = steps[step_cursor + 1]
            if next_step.get("action_type") == "tool_call":
                return "tool_caller"
        return "answer"
    elif action_type == "answer":
        return "answer"
    else:
        # Unknown action type, go to answer
        return "answer"

