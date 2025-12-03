"""Plan controller node for routing execution flow."""

import logging
from typing import Literal
from agent.state import AgentState
from agent.tracing import create_trace_event

logger = logging.getLogger(__name__)


def plan_controller_node(state: AgentState) -> AgentState:
    """
    Plan controller node: advances cursor for "think" steps and unknown action types.
    
    Responsibility: State mutation (cursor advancement).
    Routing decisions are handled by route_controller.
    """
    if not state.get("plan") or "steps" not in state.get("plan", {}):
        return state
    
    steps = state["plan"]["steps"]
    step_cursor = state.get("step_cursor", 0)
    
    if step_cursor >= len(steps):
        return state
    
    current_step = steps[step_cursor]
    action_type = current_step.get("action_type")
    
    if action_type == "think":
        warning_msg = f"Auto-advancing past 'think' step at cursor {step_cursor}"
        logger.warning(f"[CONTROLLER] {warning_msg}")
        create_trace_event(
            node_name="plan_controller",
            event_type="warning",
            state=state,
            input_data={"step_cursor": step_cursor, "action_type": action_type},
            output_data={"message": warning_msg},
        )
        state["step_cursor"] = step_cursor + 1
    
    elif action_type not in ("think", "tool_call", "answer"):
        warning_msg = f"Unknown action_type '{action_type}' at cursor {step_cursor} - auto-advancing"
        logger.warning(f"[CONTROLLER] {warning_msg}")
        create_trace_event(
            node_name="plan_controller",
            event_type="warning",
            state=state,
            input_data={"step_cursor": step_cursor, "action_type": action_type},
            output_data={"message": warning_msg},
        )
        state["step_cursor"] = step_cursor + 1
    
    return state


def route_controller(state: AgentState) -> Literal["tool_caller", "planner", "answer", "end"]:
    """
    Routing function: decides which node to execute next based on plan state.
    
    Returns:
    - "tool_caller": Execute next tool_call step
    - "planner": Replan if all steps done but not ready to answer
    - "answer": Generate final answer
    - "end": Episode complete
    """
    if state.get("done", False):
        return "end"
    
    if not state.get("plan") or "steps" not in state.get("plan", {}):
        state["done"] = True
        return "answer"
    
    steps = state["plan"]["steps"]
    step_cursor = state.get("step_cursor", 0)
    
    # All plan steps executed - check if ready to answer
    if step_cursor >= len(steps):
        if state.get("is_ready_for_answer", False):
            state["done"] = True
            return "end"
        
        # Not ready - check turn limit before replanning
        replan_count = state.get("replan_count", 0)
        max_replans = state.get("config", {}).get("max_replans", 3)
        
        if replan_count >= max_replans:
            warning_msg = f"Replan count ({replan_count}) exceeded max_replans ({max_replans}) - forcing answer"
            logger.warning(f"[CONTROLLER] {warning_msg}")
            create_trace_event(
                node_name="plan_controller",
                event_type="warning",
                state=state,
                input_data={"replan_count": replan_count, "max_replans": max_replans},
                output_data={"message": warning_msg},
            )
            state["is_ready_for_answer"] = True
            state["done"] = True
            return "answer"
        
        state["replan_count"] = replan_count + 1
        return "planner"
    
    # Still executing plan steps
    current_step = steps[step_cursor]
    action_type = current_step.get("action_type")
    
    if action_type == "tool_call":
        return "tool_caller"
    
    elif action_type == "answer":
        state["done"] = True
        return "answer"
    
    else:
        # Unknown action type - should have been auto-advanced by plan_controller_node
        warning_msg = f"Unexpected action_type '{action_type}' found by route controller at cursor {step_cursor} - routing to answer"
        logger.warning(f"[CONTROLLER] {warning_msg}")
        create_trace_event(
            node_name="plan_controller",
            event_type="warning",
            state=state,
            input_data={"step_cursor": step_cursor, "action_type": action_type},
            output_data={"message": warning_msg},
        )
        state["done"] = True
        return "answer"
