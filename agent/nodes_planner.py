"""Planner node for creating structured execution plans."""

import json
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chat_models import init_chat_model

from agent.state import AgentState, build_memory_view
from agent.tracing import traced_llm_call, create_trace_event


async def planner_node(state: AgentState) -> AgentState:
    """
    Planner node: creates a structured plan from the user query.
    
    Builds memory view, calls LLM to generate a plan, and stores it in state.plan.
    """
    # Build memory view for context
    memory_view = build_memory_view(state)
    
    # System prompt for plan generation
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
        # Initialize LLM (use planner_model from config, fallback to env, then default)
        import os
        config = state.get("config", {})
        model_name = config.get("planner_model") or os.getenv("PLANNER_MODEL") or config.get("model_name", "gpt-4o-mini")
        llm = init_chat_model(model_name)
        
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
            llm_input=messages
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

