"""Reflection node for analyzing the episode."""

import json
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chat_models import init_chat_model

from agent.state import AgentState, build_memory_view
from agent.tracing import traced_llm_call


async def reflection_node(state: AgentState) -> AgentState:
    """
    Reflection node: analyzes the episode for mistakes or improvements.
    
    Only runs if state.ready_for_reflection is True.
    """
    if not state.get("ready_for_reflection", False):
        return state
    
    # Build memory view focusing on the full episode
    memory_view = build_memory_view(state)
    
    # System prompt for structured reflection
    system_prompt = """You are a reflection assistant that analyzes agent execution episodes.

Review the conversation and tool execution history. Identify specific problems that can be fixed in a replan.

Return a JSON object with this structure:
{
  "overall_assessment": "success" | "partial_success" | "failure",
  "specific_problems": [
    {
      "type": "planning_error" | "sql_error" | "tool_parameter_error" | "tool_chaining_error" | "data_extraction_error" | "other",
      "step_id": <step number or null>,
      "tool_name": "<tool name or null>",
      "description": "Specific description of the problem",
      "details": "Additional context about what went wrong",
      "suggested_fix": "What should be done differently"
    }
  ],
  "plan_issues": [
    {
      "step_id": <step number>,
      "issue": "Description of what was wrong with this step",
      "correct_action": "What should have been done instead"
    }
  ],
  "sql_issues": [
    {
      "generated_sql": "<the SQL that was generated>",
      "problem": "What was wrong with it (wrong table, wrong column, wrong logic, etc.)",
      "correct_sql": "<what the SQL should have been, if known>"
    }
  ],
  "tool_parameter_errors": [
    {
      "tool_name": "<tool name>",
      "wrong_parameters": {"param": "value"},
      "correct_parameters": {"param": "value"},
      "reason": "Why the parameters were wrong"
    }
  ],
  "tool_chaining_issues": [
    {
      "from_tool": "<tool name>",
      "to_tool": "<tool name>",
      "problem": "What data was not correctly passed between tools",
      "missing_data": "What data should have been extracted and passed"
    }
  ],
  "general_feedback": "Overall feedback on the execution"
}

Focus on actionable problems that can be fixed in a replan. Be specific about:
- Wrong SQL queries (table names, column names, logic errors)
- Wrong tool parameters (amounts, currencies, query strings)
- Missing data extraction from tool results
- Incorrect tool chaining (data not passed correctly between tools)
- Planning mistakes (wrong tool selected, wrong order, missing steps)"""

    # Build tool results summary for reflection
    tool_results_summary = []
    for tr in state.get("tool_results", []):
        tool_results_summary.append({
            "step_id": tr.get("step_id"),
            "tool_name": tr.get("tool_name"),
            "input": tr.get("raw_input"),
            "output_summary": str(tr.get("raw_output"))[:500] if tr.get("raw_output") else None,
        })
    
    user_prompt = f"""Review this agent execution episode:

User query: {state["user_query"]}

Final answer: {state.get("answer_draft", "N/A")}

Plan executed:
{json.dumps(state.get("plan", {}), indent=2) if state.get("plan") else "No plan"}

Tool execution results:
{json.dumps(tool_results_summary, indent=2) if tool_results_summary else "No tool results"}

Execution history:
{memory_view}

Analyze and return structured JSON identifying specific problems."""

    try:
        # Initialize LLM (use reflection_model from config, fallback to env, then default)
        import os
        config = state.get("config", {})
        model_name = config.get("reflection_model") or os.getenv("REFLECTION_MODEL") or config.get("model_name", "gpt-4o-mini")
        llm = init_chat_model(model_name)
        
        # Prepare messages
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        # Call LLM with tracing
        response = await traced_llm_call(
            node_name="reflection",
            state=state,
            llm_callable=llm,
            llm_input=messages
        )
        
        # Parse structured reflection JSON
        reflection_text = response.content.strip()
        if reflection_text.startswith("```"):
            lines = reflection_text.split("\n")
            reflection_text = "\n".join(lines[1:-1]) if len(lines) > 2 else reflection_text
            reflection_text = reflection_text.replace("```json", "").replace("```", "").strip()
        
        try:
            reflection_data = json.loads(reflection_text)
            # Store structured reflection in state for potential replanning
            state["reflection"] = reflection_data
        except json.JSONDecodeError:
            # Fallback: store as text if JSON parsing fails
            state["reflection"] = {"raw_text": reflection_text}
        
    except Exception as e:
        # On error, just mark as done
        pass
    
    # Mark episode as complete
    state["done"] = True
    
    return state

