"""Planner node for creating structured execution plans."""

import json
import copy
from langchain_core.messages import HumanMessage, SystemMessage
from agent.state import AgentState, build_memory_view
from agent.tracing import traced_llm_call, create_trace_event
from agent.llm_utils import init_llm


async def planner_node(state: AgentState) -> AgentState:
    """
    Planner node: creates a structured plan from the user query.
    
    Two modes:
    1. Initial planning: Creates the first plan.
    2. Replanning: Analyzes previous execution, adds correction steps if needed.
    """
    # Check if we are replanning (plan already exists)
    is_replanning = state.get("plan") is not None and "steps" in state.get("plan", {})
    
    # Build memory view for context
    memory_view = build_memory_view(state)
    
    # Initialize LLM
    import os
    config = state.get("config", {})
    model_name = config.get("planner_model") or os.getenv("PLANNER_MODEL") or config.get("model_name", "gpt-4o-mini")
    llm, actual_model_name = init_llm(model_name)
    
    if is_replanning:
        return await _replan(state, llm, memory_view, actual_model_name)
    else:
        return await _initial_plan(state, llm, memory_view, actual_model_name)


async def _initial_plan(state: AgentState, llm, memory_view: str, model_name: str) -> AgentState:
    """Generate the initial plan."""
    system_prompt = """You are a planning assistant that breaks down user queries into structured execution plans.

Given a user query and the conversation history, create a step-by-step plan to answer it.

Output a JSON object with this structure:
{
  "steps": [
    {
      "id": 1,
      "description": "Step description",
      "action_type": "tool_call" | "think" | "answer",
      "tool": "sql_tool" | "rag_tool" | "api_tool" | null,
      "query": "Specific query for this step (required for sql_tool steps)"
    }
  ]
}

IMPORTANT: For sql_tool steps, you MUST include a "query" field with a specific, focused natural language query for that step.
- If you need multiple SQL queries (e.g., "count invoices" and "average invoice amount"), create separate steps with different queries
- Each sql_tool step should have ONE specific query, not the full user question
- Example: If the user asks "How many invoices and what's the average?", create:
  - Step 1: {"tool": "sql_tool", "query": "How many invoices are there?"}
  - Step 2: {"tool": "sql_tool", "query": "What is the average invoice amount?"}

Action types:
- "tool_call": Execute a tool (requires "tool" field)
- "think": Internal reasoning step (no tool needed)
- "answer": Generate final answer (no tool needed)

Available tools and when to use them:

1. sql_tool (Database Query Tool):
   Use for questions about DATA in the Chinook music store database:
   - Sales data, revenue, quantities, counts
   - Customer information, employee data
   - Track/album/artist information
   - Invoice and invoice line details
   - Aggregations, sums, averages, counts
   - Filtering by date, genre, artist, etc.
   Examples: "How many tracks in Latin genre?", "Total revenue in 2013", "Which artist has most albums?"
   
   Note: The sql_tool automatically includes the database schema (table names, columns, relationships) 
   when generating SQL queries, so you don't need a separate step to fetch the schema.

2. rag_tool (Document Retrieval Tool):
   Use for questions about KNOWLEDGE, POLICIES, or GENERAL INFORMATION stored in documents:
   - Company information and business details
   - Pricing policies and structures
   - Currency conversion rates and policies
   - Store policies and procedures
   - General knowledge about the music store
   - Information that is NOT in the database but in knowledge documents
   
   The document store contains:
   - music_store_info.txt: General company information, business model, founding date
   - pricing_policy.txt: Pricing structure, currency conversion rates (EUR/USD, GBP/USD, etc.)
   - genre_information.txt: Information about music genres, genre statistics
   
   Examples: "What is the currency conversion rate?", "What is the store's pricing policy?", 
   "What currencies does the store accept?", "When was the store founded?"

3. api_tool (API Tool):
   Use for:
   - Currency conversion calculations (USD to EUR, GBP, JPY, CAD, AUD and vice versa)
   - Weather information (placeholder)
   - Other external API operations
   
   Examples: "Convert 100 USD to EUR", "What is the weather in San Francisco?"
   
   Note: If a query asks about currency conversion AND mentions "based on store's currency policy",
   you should FIRST use rag_tool to retrieve the policy/rates, THEN use api_tool to perform the conversion.

Planning strategy:
- If a query needs DATA from the database → use sql_tool
- If a query needs KNOWLEDGE/POLICY information → use rag_tool
- If a query needs to PERFORM a calculation/operation → use api_tool
- If a query needs multiple pieces of information, plan multiple tool calls in sequence
- Always end with an "answer" step to synthesize the results

Return ONLY valid JSON, no markdown, no explanation."""

    user_prompt = f"""User query: {state["user_query"]}

Conversation history:
{memory_view}

Create a plan to answer the user's query. Output the JSON plan."""

    try:
        # Prepare messages
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        # Call LLM with tracing
        response = await traced_llm_call(
            node_name="planner",
            state=state,
            llm_callable=llm,
            llm_input=messages,
            model_name=model_name
        )
        
        # Parse the plan from response
        plan_text = response.content.strip()
        
        # Remove markdown code blocks if present
        if plan_text.startswith("```"):
            lines = plan_text.split("\n")
            plan_text = "\n".join(lines[1:-1]) if len(lines) > 2 else plan_text
            plan_text = plan_text.replace("```json", "").replace("```", "").strip()
        
        # Parse JSON
        plan = json.loads(plan_text)
        
        # Validate plan structure
        if not isinstance(plan, dict) or "steps" not in plan:
            raise ValueError("Plan must have a 'steps' key")
        
        if not isinstance(plan["steps"], list):
            raise ValueError("Plan steps must be a list")
        
        # Store plan and reset cursor
        state["plan"] = plan
        state["step_cursor"] = 0
        state["is_ready_for_answer"] = False  # Reset readiness
        
        # Log plan update
        create_trace_event(
            node_name="planner",
            event_type="plan_update",
            state=state,
            input_data={"user_query": state["user_query"], "memory_view": memory_view},
            output_data={"plan": plan},
        )
        
    except Exception as e:
        # On error, create a simple fallback plan
        state["plan"] = {
            "steps": [
                {
                    "id": 1,
                    "description": f"Answer the query: {state['user_query']}",
                    "action_type": "answer",
                    "tool": None
                }
            ]
        }
        state["step_cursor"] = 0
        
        create_trace_event(
            node_name="planner",
            event_type="plan_update",
            state=state,
            input_data={"user_query": state["user_query"]},
            output_data={"plan": state["plan"], "error": str(e)},
            error=str(e),
        )
    
    return state


async def _replan(state: AgentState, llm, memory_view: str, model_name: str) -> AgentState:
    """Analyze execution and update plan if needed."""
    
    # Increment replan count - this is called when we need to replan
    state["replan_count"] = state.get("replan_count", 0) + 1
    
    system_prompt = """You are a sophisticated planning assistant that reviews agent execution.
    
Your goal is to check if the current plan execution has gathered enough correct information to answer the user's query.
The first time you are called, there will be just a user query, no plan or execution history, so you should just plan the steps to answer the user's query.

Review the:
1. User Query
2. Execution History (tool inputs/outputs/errors)
3. Current Plan

Decide:
- Is the information sufficient and correct?
- Did any tool call fail or return incorrect data?
- Do we need to add new steps to fix errors or get missing info?

Output a JSON object:
{
  "is_ready_for_answer": boolean,
  "reasoning": "Explanation of your decision",
  "feedback": "Critique or feedback for the next tool execution (if needed), e.g., 'The last SQL query failed because table X doesn't exist'",
  "new_steps": [
    // Optional: Only include if is_ready_for_answer is false
    {
      "id": 1,
      "description": "Fix step description",
      "action_type": "tool_call",
      "tool": "...",
      "query": "..."
    }
  ]
}

If "is_ready_for_answer" is true, "new_steps" should be empty.
If "is_ready_for_answer" is false, provide "new_steps" to append to the plan.
The "feedback" field is crucial for helping downstream tools correct their mistakes.
"""

    user_prompt = f"""User query: {state["user_query"]}

Current Plan Status:
{json.dumps(state["plan"], indent=2)}

Execution History:
{memory_view}

Analyze the execution. Are we ready to answer? If not, provide new steps and feedback."""

    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = await traced_llm_call(
            node_name="planner_replan",
            state=state,
            llm_callable=llm,
            llm_input=messages,
            model_name=model_name
        )
        
        result_text = response.content.strip()
        if result_text.startswith("```"):
            lines = result_text.split("\n")
            result_text = "\n".join(lines[1:-1]) if len(lines) > 2 else result_text
            result_text = result_text.replace("```json", "").replace("```", "").strip()
            
        result = json.loads(result_text)
        
        is_ready = result.get("is_ready_for_answer", False)
        new_steps = result.get("new_steps", [])
        feedback = result.get("feedback", "")
        
        # If planner says not ready but provides no new steps, force readiness to prevent infinite loop
        if not is_ready and not new_steps:
            is_ready = True
        
        state["is_ready_for_answer"] = is_ready
        state["feedback"] = feedback
        
        if not is_ready and new_steps:
            # Snapshot current plan before modification
            current_plan = copy.deepcopy(state["plan"])
            state.setdefault("plan_history", []).append(current_plan)
            
            steps = state["plan"]["steps"]
            
            # We will append new steps at the end, and then jump the cursor to the first new step.
            # This ensures they actually get executed next.
            insertion_index = len(steps)  # index of the first new step after append
            
            # Ensure IDs are unique/sequential based on previous last step
            last_id = 0
            if steps:
                last_step = steps[-1]
                last_id = last_step.get("id", 0)
                if isinstance(last_id, str):
                    try:
                        last_id = int(last_id)
                    except:
                        last_id = 0

            for i, step in enumerate(new_steps):
                step["id"] = last_id + i + 1
                steps.append(step)
            
            # IMPORTANT: move cursor to the first new step
            state["step_cursor"] = insertion_index
            
        create_trace_event(
            node_name="planner_replan",
            event_type="replan",
            state=state,
            input_data={"memory_view": memory_view},
            output_data={"is_ready": is_ready, "new_steps_count": len(new_steps), "feedback": feedback},
        )

    except Exception as e:
        # Fallback: assume ready to answer to prevent infinite loops on error
        state["is_ready_for_answer"] = True
        create_trace_event(
            node_name="planner_replan",
            event_type="error",
            state=state,
            error=str(e)
        )

    return state
