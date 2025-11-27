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
    # For "answer" steps, advance cursor if we're ready (to prevent re-checking)
    if state.get("plan") and "steps" in state.get("plan", {}):
        steps = state["plan"]["steps"]
        step_cursor = state.get("step_cursor", 0)
        
        if step_cursor < len(steps):
            current_step = steps[step_cursor]
            action_type = current_step.get("action_type")
            
            # For "think" steps, advance cursor
            if action_type == "think":
                state["step_cursor"] = step_cursor + 1
            # For "answer" steps, if we're ready, advance cursor to prevent re-checking
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
    # Check if done
    if state.get("done", False):
        return "end"
    
    # Check if we have a plan
    if not state.get("plan") or "steps" not in state.get("plan", {}):
        # No plan, go to answer (or planner? but usually answer if planner failed)
        return "answer"
    
    steps = state["plan"]["steps"]
    step_cursor = state.get("step_cursor", 0)
    
    # Check if we've completed all steps
    if step_cursor >= len(steps):
        # All steps done. Check if we are ready for answer or need to replan.
        if state.get("is_ready_for_answer", False):
            return "answer"
        else:
            # Check replan limit
            replan_count = state.get("replan_count", 0)
            max_replans = state.get("config", {}).get("max_replans", 3)
            
            if replan_count >= max_replans:
                return "answer"
                
            # Not ready? Go back to planner to check results and potentially replan
            return "planner"
    
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
        # If think is the last step, we fall through to "all steps done" logic
        # But here we are inside the cursor < len(steps) block.
        # So we should probably check readiness or continue?
        # For simplicity, if next is answer or end of list, we loop back to planner/answer via next iteration
        # effectively handling it by re-entering this function with incremented cursor.
        # But since this function is edge logic, we need to decide WHERE to go.
        # If 'think' was the current step, the node logic already incremented cursor?
        # NO. The node logic increments cursor for 'think'.
        # The routing logic looks at the *new* cursor?
        # Let's check plan_controller_node.
        pass # Logic handled by node incrementing cursor
        
        # If the node incremented the cursor, we need to check the NEW cursor state.
        # But route_controller receives the state *after* plan_controller_node ran?
        # Yes.
        
        # So if we are here, it means the cursor is pointing to a step (because of the initial check).
        # Wait, if node incremented cursor, and it is now >= len, we wouldn't be in this block?
        # Correct. If step_cursor >= len(steps), we hit the first block.
        # So if we are here, there IS a current step.
        return "answer" # Fallback, should not happen for "think" if logic is correct
        
    elif action_type == "answer":
        # For "answer" steps, we need to check if we should replan BEFORE executing the answer
        # BUT only if we've actually executed some tool calls (not just a fallback plan)
        # Check if we have any tool results - if not, this is likely a fallback plan, so just answer
        has_tool_results = bool(state.get("tool_results")) or bool(state.get("sql_results")) or bool(state.get("rag_docs")) or bool(state.get("api_results"))
        
        if has_tool_results and not state.get("is_ready_for_answer", False):
            # We've executed tools but planner hasn't confirmed readiness - check replanning
            replan_count = state.get("replan_count", 0)
            max_replans = state.get("config", {}).get("max_replans", 3)
            
            if replan_count < max_replans:
                # Not ready and haven't hit limit - go to planner for re-evaluation
                return "planner"
        # Ready for answer, hit replan limit, or no tools executed (fallback plan) - proceed to answer
        # Note: Cursor will be advanced in plan_controller_node if is_ready_for_answer is True
        return "answer"
    else:
        # Unknown action type, go to answer
        return "answer"

