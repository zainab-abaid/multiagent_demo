"""Agent state and trace event models."""

from dataclasses import dataclass, field, asdict
from typing import Optional, TypedDict


@dataclass
class AgentConfig:
    """Configuration for the agent."""
    model_name: str = "gpt-4o-mini"  # Default model (can be overridden by env)
    planner_model: str = "gpt-4o-mini"
    sql_model: str = "gpt-4o-mini"
    answer_model: str = "gpt-4o-mini"
    reflection_model: str = "gpt-4o-mini"
    judge_model: str = "gpt-4o-mini"
    use_compression: bool = False
    compression_strategy: str = "raw"  # "raw", "turn_summary", "phase_summary", "structured_json"
    max_replans: int = 3  # Maximum number of replanning attempts allowed


@dataclass
class TraceEvent:
    """A single event in the agent's trajectory."""
    event_id: str
    timestamp: float
    node_name: str  # e.g. "planner", "tool_caller", "sql_tool", "rag_tool"
    event_type: str  # "llm_call", "tool_call", "plan_update", "reflection", "episode_start", etc.
    
    # Core payloads
    input: Optional[dict | str] = None
    output: Optional[dict | str] = None
    
    # Metrics
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    latency_ms: Optional[float] = None
    
    # Optional details
    tool_name: Optional[str] = None  # for tool calls
    model_name: Optional[str] = None  # actual model used for LLM calls
    error: Optional[str] = None  # error message if any


# TypedDict for LangGraph compatibility
class AgentState(TypedDict):
    """The state of the agent during an episode (TypedDict for LangGraph)."""
    user_query: str
    plan: Optional[dict]  # current plan object, structured (e.g. dict with list of steps)
    step_cursor: int  # which step of the plan we are at
    trajectory: list[dict]  # full log of everything that happened (stored as dicts for serialization)
    memory_view: Optional[str | dict]  # what planner / reflection sees (raw or summarized)
    answer_draft: Optional[str]
    ready_for_reflection: bool
    done: bool
    config: dict  # AgentConfig stored as dict
    latest_result: Optional[dict | str]  # store latest tool result for convenience
    tool_results: list[dict]  # structured list of tool execution results
    reflection: Optional[dict]  # structured reflection output for replanning
    sql_results: Optional[list]  # list of SQL query results: [{generated_sql, query_result, numeric_values, step_query}, ...]
    rag_docs: Optional[list]  # list of RAG document dicts
    api_results: Optional[list]  # list of API call results: [{tool, input, output}, ...]
    plan_history: list[dict]  # snapshots of previous plans before replanning
    feedback: Optional[str]  # feedback from planner to tool execution
    is_ready_for_answer: bool  # signal from planner that enough info is gathered
    replan_count: int  # count of replanning attempts


def create_initial_state(user_query: str, config: Optional[AgentConfig] = None) -> AgentState:
    """Create an initial AgentState from a user query."""
    import os
    
    if config is None:
        config = AgentConfig()
    
    # Always override with environment variables if set (even if config was provided)
    config.planner_model = os.getenv("PLANNER_MODEL", config.planner_model)
    config.sql_model = os.getenv("SQL_MODEL", config.sql_model)
    config.answer_model = os.getenv("ANSWER_MODEL", config.answer_model)
    config.reflection_model = os.getenv("REFLECTION_MODEL", config.reflection_model)
    config.judge_model = os.getenv("JUDGE_MODEL", config.judge_model)
    # Also set default model_name if not set
    config.model_name = os.getenv("DEFAULT_MODEL", config.model_name)
    
    # Max replans
    config.max_replans = int(os.getenv("MAX_REPLANS", config.max_replans))
    
    return AgentState(
        user_query=user_query,
        plan=None,
        step_cursor=0,
        trajectory=[],
        memory_view=None,
        answer_draft=None,
        ready_for_reflection=False,
        done=False,
        config=asdict(config),
        latest_result=None,
        tool_results=[],
        reflection=None,
        sql_results=None,
        rag_docs=None,
        api_results=None,
        plan_history=[],
        feedback=None,
        is_ready_for_answer=False,
        replan_count=0,
    )


def event_to_dict(event: TraceEvent) -> dict:
    """Convert TraceEvent to dict for storage in state."""
    return asdict(event)


def dict_to_event(event_dict: dict) -> TraceEvent:
    """Convert dict back to TraceEvent."""
    return TraceEvent(**event_dict)


def build_memory_view(state: AgentState) -> str:
    """
    Build a memory view from the trajectory for planner/reflection nodes.
    
    For now: simple raw concat of trajectory summaries.
    Later: can be replaced with real compression strategies.
    """
    if not state.get("trajectory"):
        return "No events yet."
    
    # Simple approach: summarize last N events
    # trajectory is stored as list of dicts
    recent_events = state["trajectory"][-20:]  # last 20 events
    
    parts = []
    for event_dict in recent_events:
        node_name = event_dict.get("node_name", "unknown")
        event_type = event_dict.get("event_type", "unknown")
        event_summary = f"[{node_name}:{event_type}]"
        
        if event_dict.get("input"):
            input_str = str(event_dict["input"])[:200]  # truncate long inputs
            event_summary += f" Input: {input_str}"
        if event_dict.get("output"):
            output_str = str(event_dict["output"])[:200]  # truncate long outputs
            event_summary += f" Output: {output_str}"
        if event_dict.get("error"):
            event_summary += f" Error: {event_dict['error']}"
        parts.append(event_summary)
    
    return "\n".join(parts)

