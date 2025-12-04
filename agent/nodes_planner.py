"""Planner node for creating structured execution plans."""

import json
import copy
from langchain_core.messages import HumanMessage, SystemMessage
from agent.state import AgentState, build_memory_view
from agent.tracing import traced_llm_call, create_trace_event
from agent.llm_utils import init_llm


# Shared tool descriptions (used in both initial plan and replan prompts)
_TOOL_DESCRIPTIONS = """
1. sql_tool (Database Query Tool):
   Use for DATA queries in the Chinook music store database:
   - Sales data, revenue, quantities, counts
   - Customer/employee information
   - Track/album/artist DATA (counts, IDs, relationships - NOT descriptions)
   - Invoice and invoice line details
   - Aggregations, sums, averages, counts
   - Filtering by date, genre, artist, etc.
   
   Examples: "How many tracks in Latin genre?", "Total revenue in 2013", 
   "Which artist has most albums?", "Average tracks per album for Body Count"
   
   Note: Database schema is automatically included when generating SQL queries.

2. rag_tool (Document Retrieval Tool):
   Use for KNOWLEDGE, POLICIES, and GENERAL INFORMATION from documents:
   - Company information and business details
   - Pricing policies and structures
   - Currency conversion rates and policies
   - Store policies and procedures
   - Artist information (background, descriptions, history - NOT data like counts)
   - Information NOT in the database but in knowledge documents
   
   Documents: music_store_info.txt, pricing_policy.txt, genre_information.txt, artist_information.txt
   
   Examples: "What is the USD to EUR exchange rate?", "What is the store's pricing policy?",
   "When was the store founded?", "What information does the store have about Body Count?"

3. api_tool (API Tool):
   Use for calculations and operations requiring tool execution:
   - Currency conversion: convert_currency_from_usd, convert_currency_to_usd
   - Value calculations: calculate_total_value (quantity × price), calculate_estimated_revenue (count × average)
   - Formatting: format_duration_hours (minutes → hours:minutes)
   - Math: calculate_percentage (part/total × 100)
   
   The api_router automatically selects the appropriate tool based on your query.
   
   Examples: "Convert 100 USD to EUR", "Calculate total value from 10 tracks at $1.05 each",
   "Convert 128 minutes to hours and minutes"
"""


_INITIAL_PLAN_PROMPT = f"""You are a planning assistant that breaks down user queries into structured execution plans.

Output a JSON object with this structure:
{{
  "steps": [
    {{
      "id": 1,
      "description": "Step description",
      "action_type": "tool_call" | "think" | "answer",
      "tool": "sql_tool" | "rag_tool" | "api_tool" | null,
      "query": "Specific query for this step (required for sql_tool and rag_tool steps)"
    }}
  ]
}}

CRITICAL RULES:
- action_type must be "tool_call", "think", or "answer" - NOT a tool name
- For sql_tool and rag_tool steps, include a "query" field with a focused, specific query
- Do NOT copy the full user query - create focused queries for each step
- Each tool step should have ONE specific query, not the full user question

Query Examples:
- User: "How many invoices and what's the average?"
  → Step 1: {{"tool": "sql_tool", "query": "How many invoices are there?"}}
  → Step 2: {{"tool": "sql_tool", "query": "What is the average invoice amount?"}}
- User: "What is the total value in EUR based on store policy?"
  → Step 1: {{"tool": "sql_tool", "query": "What is the total value in USD?"}}
  → Step 2: {{"tool": "rag_tool", "query": "What is the USD to EUR exchange rate in the store's pricing policy?"}}
  → Step 3: {{"tool": "api_tool", "query": "Convert the total USD value to EUR using the rate from RAG"}}

Available tools:
{_TOOL_DESCRIPTIONS}

Planning Strategy:
- DATA queries (counts, averages, revenue) → sql_tool
- KNOWLEDGE/POLICY queries (descriptions, company info, policies) → rag_tool
- CALCULATIONS/OPERATIONS (conversions, math, formatting) → api_tool
- Artist queries: sql_tool for DATA (counts, IDs), rag_tool for KNOWLEDGE (descriptions, background)
- Multiple information needs → plan multiple tool calls in sequence
- Always end with an "answer" step

Return ONLY valid JSON, no markdown, no explanation."""


_REPLAN_PROMPT = f"""You are a planning assistant that reviews agent execution and proposes corrections.

IMPORTANT: Check the "Replanning state" section in the memory view to see:
- Current replanning attempt number (e.g., "This is replanning attempt 2/6")
- What queries have already been executed (see RAG TOOL EXECUTIONS and SQL TOOL EXECUTIONS)

CRITICAL RULES:
1. DO NOT repeat queries that have already been executed - check the execution history first
2. If a RAG query was already tried and returned documents, do NOT request the same query again
3. If information doesn't exist in the documents after trying, set "is_ready_for_answer": true and proceed
4. If the same query keeps returning the same results, accept that the information isn't available

Your goal: Check if execution gathered enough correct information to answer the user's query.
If not, propose concrete new TOOL CALL steps to fix errors or get missing information.

Available tools:
{_TOOL_DESCRIPTIONS}

Review:
1. User Query
2. Execution History (tool inputs/outputs/errors from memory_view) - PAY ATTENTION TO WHAT QUERIES WERE ALREADY TRIED
3. Current Plan - check if steps were already executed

Decide:
- Is information sufficient and correct?
- Did any tool call fail or return incorrect data?
- Is needed information missing?
- What new steps are needed to fix errors or get missing info?
- HAVE WE ALREADY TRIED THIS QUERY? (Check RAG/SQL execution history)

Currency Conversion Policy:
- If user query mentions "based on store's policy" and RAG documents contain exchange rates
  (even if marked "approximate"), treat those rates as AUTHORITATIVE
- Do NOT request another RAG query if rates are already present - proceed directly to api_tool conversion
- Only request a new RAG query if NO rates were found at all

New Steps Guidelines:
- If information is missing and requires TOOLS, add "tool_call" steps (NOT "think")
- sql_tool: for missing database data (prices, counts, averages)
- rag_tool: for document retrieval. IMPORTANT: 
  * Create SPECIFIC queries, not the full user query
  * Example: Use "What is the USD to EUR exchange rate?" not the full user question
  * CHECK THE EXECUTION HISTORY FIRST - do NOT request a query that was already executed
  * If a similar query was already tried and returned documents, those documents contain all available information
- api_tool: for calculations, conversions, formatting, math operations

"think" steps are ONLY for:
- Re-interpreting existing tool results already in the right format
- Simple reasoning without calculations

"think" steps MUST NOT be used for:
- Calculations (use api_tool: format_duration_hours, calculate_total_value, etc.)
- Unit conversions (use api_tool: format_duration_hours, convert_currency_from_usd, etc.)
- Computing percentages or values (use api_tool: calculate_percentage, calculate_total_value, etc.)

CRITICAL: If you need to convert minutes to hours, calculate totals, convert currencies, or do any math,
you MUST use api_tool with an appropriate query, NOT a "think" step.

Alternative Strategies:
- If document query failed to give numeric value → consider sql_tool to compute from database
- If exchange rates retrieved from RAG → use api_tool to perform conversion
- Avoid multiple "think" steps that don't introduce new TOOL outputs

Output JSON:
{{
  "is_ready_for_answer": boolean,
  "reasoning": "Explanation of your decision",
  "feedback": "Critique for next tool execution (if needed)",
  "new_steps": [
    {{
      "id": 1,
      "description": "Step description",
      "action_type": "tool_call" | "think",
      "tool": "sql_tool" | "rag_tool" | "api_tool" | null,
      "query": "Specific query (required for sql_tool and rag_tool steps)"
    }}
  ]
}}

CRITICAL: action_type must be "tool_call", "think", or "answer" - NOT a tool name.
If using a tool, set action_type to "tool_call" and put tool name in "tool" field.

If "is_ready_for_answer" is true, "new_steps" should be empty.
If required information truly doesn't exist (e.g., RAG queries already tried and returned no relevant info), 
set "is_ready_for_answer" to true and explain why in the reasoning field.

STOP REPEATING QUERIES: If you see the same RAG query in the execution history multiple times with the same results,
it means the information isn't available. Set "is_ready_for_answer": true and proceed to answer with available information."""


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
    user_prompt = f"""User query: {state["user_query"]}

Conversation history:
{memory_view}

Create a plan to answer the user's query. Output the JSON plan."""

    try:
        messages = [
            SystemMessage(content=_INITIAL_PLAN_PROMPT),
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
    
    user_prompt = f"""User query: {state["user_query"]}

Current Plan Status:
{json.dumps(state["plan"], indent=2)}

Execution History:
{memory_view}

Analyze the execution. Are we ready to answer? If not, provide new steps and feedback."""

    try:
        messages = [
            SystemMessage(content=_REPLAN_PROMPT),
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
        
        # Handle edge cases to prevent infinite loops
        # If planner says ready, ignore any new_steps (shouldn't happen but be safe)
        if is_ready:
            state["is_ready_for_answer"] = True
            state["feedback"] = feedback
        # If planner says not ready but provides no new steps, force readiness to prevent infinite loop
        elif not new_steps:
            state["is_ready_for_answer"] = True
            state["feedback"] = feedback or "Planner indicated not ready but provided no new steps - forcing readiness"
        # If not ready and has new steps, add them
        else:
            state["is_ready_for_answer"] = False
            state["feedback"] = feedback
            # Snapshot current plan before modification
            current_plan = copy.deepcopy(state["plan"])
            state.setdefault("plan_history", []).append(current_plan)
            
            steps = state["plan"]["steps"]
            
            # Append new steps at the end
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
            
            # Move cursor to the first new step so they get executed next
            state["step_cursor"] = insertion_index
        
        create_trace_event(
            node_name="planner_replan",
            event_type="replan",
            state=state,
            input_data={"memory_view": memory_view},
            output_data={"is_ready": is_ready, "new_steps": new_steps, "new_steps_count": len(new_steps), "feedback": feedback},
        )

    except Exception as e:
        # Fallback: assume ready to answer to prevent infinite loops on error
        state["is_ready_for_answer"] = True
        create_trace_event(
            node_name="planner_replan",
            event_type="error",
            state=state,
            input_data={"memory_view": memory_view},
            output_data={"error": str(e)},
            error=str(e),
        )
    
    return state
