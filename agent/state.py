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
    sql_results: Optional[list]  # list of SQL query results: [{step_id, generated_sql, query_result, step_query, success}, ...]
    rag_results: Optional[list]  # list of RAG query results: [{step_id, query, docs}, ...]
    api_results: Optional[list]  # list of API call results: [{step_id, tool, input, output}, ...]
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
        rag_results=None,
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
    Build a structured memory view from state for planner/reflection nodes.
    
    Creates clean, structured summaries of tool executions instead of raw trajectory logs.
    This makes it easier for LLMs to understand what happened and reduces hallucination.
    """
    parts = []
    
    # Add context header
    parts.append("=== CONTEXT ===")
    parts.append(f'User query: "{state.get("user_query", "N/A")}"')
    
    # Current plan summary
    plan = state.get("plan")
    if plan and "steps" in plan:
        parts.append("\nCurrent plan:")
        for step in plan.get("steps", []):
            step_id = step.get("id", "?")
            action_type = step.get("action_type", "unknown")
            tool = step.get("tool", "")
            description = step.get("description", "")
            tool_str = f"/{tool}" if tool else ""
            parts.append(f"  - Step {step_id} ({action_type}{tool_str}): \"{description}\"")
    else:
        parts.append("\nCurrent plan: No plan yet")
    
    # Replanning state
    replan_count = state.get("replan_count", 0)
    max_replans = state.get("config", {}).get("max_replans", 3)
    parts.append(f"\nReplanning state:")
    parts.append(f"  - replan_count: {replan_count}")
    parts.append(f"  - max_replans: {max_replans}")
    if replan_count > 0:
        parts.append(f"  - This is replanning attempt {replan_count + 1}/{max_replans + 1}")
    parts.append("")
    
    # Build SQL tool execution summaries
    sql_results = state.get("sql_results") or []
    if sql_results:
        parts.append("=== SQL TOOL EXECUTIONS ===")
        for sql_res in sql_results:
            step_id = sql_res.get("step_id", "N/A")
            step_query = sql_res.get("step_query", "N/A")
            generated_sql = sql_res.get("generated_sql", "N/A")
            query_result = sql_res.get("query_result")
            success = sql_res.get("success", query_result is not None)
            
            parts.append(f"\nExecuted tool: sql_tool")
            parts.append(f"  Step ID: {step_id}")
            parts.append(f"  Natural language query: {step_query}")
            parts.append(f"  SQL query: {generated_sql}")
            parts.append(f"  Status: {'success' if success else 'failed'}")
            if query_result is not None:
                # Try to parse numeric result for cleaner display
                result_str = str(query_result)
                if result_str.startswith("[(") and result_str.endswith(")]"):
                    # Parse tuple result like "[(10,)]"
                    try:
                        import ast
                        parsed = ast.literal_eval(result_str)
                        if parsed and len(parsed) > 0 and len(parsed[0]) > 0:
                            parts.append(f"  Result (parsed): {parsed[0][0]}")
                        else:
                            parts.append(f"  Result: {result_str}")
                    except:
                        parts.append(f"  Result: {result_str}")
                else:
                    parts.append(f"  Result: {result_str}")
            else:
                parts.append(f"  Result: None")
    
    # Build API tool execution summaries
    api_results = state.get("api_results") or []
    if api_results:
        parts.append("\n=== API TOOL EXECUTIONS ===")
        for api_res in api_results:
            step_id = api_res.get("step_id", "N/A")
            tool_name = api_res.get("tool", "unknown")
            input_params = api_res.get("input", {})
            output = api_res.get("output")
            error = api_res.get("error")
            
            parts.append(f"\nExecuted tool: {tool_name}")
            parts.append(f"  Step ID: {step_id}")
            parts.append(f"  Input parameters: {input_params}")
            if error:
                parts.append(f"  Status: failed")
                parts.append(f"  Error: {error}")
            else:
                parts.append(f"  Status: success")
                parts.append(f"  Output: {output}")
    
    # Build RAG tool execution summaries (show queries and results)
    rag_results = state.get("rag_results") or []
    if rag_results:
        parts.append("\n=== RAG TOOL EXECUTIONS ===")
        for rag_res in rag_results:
            step_id = rag_res.get("step_id", "N/A")
            query = rag_res.get("query", "N/A")
            docs = rag_res.get("docs", [])
            
            parts.append(f"\nExecuted tool: rag_tool")
            parts.append(f"  Step ID: {step_id}")
            parts.append(f"  Query: {query}")
            parts.append(f"  Status: success")
            parts.append(f"  Retrieved {len(docs)} document(s):")
            for idx, doc in enumerate(docs[:3], 1):  # Show first 3 docs to avoid too much text
                content = doc.get("content", "")
                score = doc.get("score")
                parts.append(f"    Document #{idx}:")
                if score is not None:
                    parts.append(f"      Relevance score: {score:.3f}")
                # Truncate content for readability
                content_preview = content[:300] + "..." if len(content) > 300 else content
                parts.append(f"      Content: {content_preview}")
            if len(docs) > 3:
                parts.append(f"    ... and {len(docs) - 3} more document(s)")
    
    if not parts:
        return "No tool executions yet."
    
    return "\n".join(parts)

