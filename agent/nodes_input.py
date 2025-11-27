"""Input node for initializing agent state."""

from agent.state import AgentState, AgentConfig
from agent.tracing import create_trace_event


def input_node(state: AgentState) -> AgentState:
    """
    Input node: initializes AgentState from a user query.
    
    Sets initial values and logs episode start.
    """
    # Initialize state fields if not already set
    state["plan"] = None
    state["step_cursor"] = 0
    if "trajectory" not in state:
        state["trajectory"] = []
    state["memory_view"] = None
    state["answer_draft"] = None
    state["ready_for_reflection"] = False
    state["done"] = False
    
    # Log episode start
    create_trace_event(
        node_name="input",
        event_type="episode_start",
        state=state,
        input_data={"user_query": state["user_query"]},
        output_data={"initialized": True},
    )
    
    return state

