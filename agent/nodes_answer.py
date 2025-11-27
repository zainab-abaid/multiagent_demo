# agent/nodes_answer.py

"""Answer generation node."""

from langchain_core.messages import HumanMessage, SystemMessage
from agent.llm_utils import init_llm

from agent.state import AgentState, build_memory_view
from agent.tracing import traced_llm_call


async def answer_node(state: AgentState) -> AgentState:
    """
    Answer node: generates the final user-facing answer.

    It uses structured tool outputs placed in state by tool_caller_node:
    - state["sql_results"]: list of SQL query results: [{generated_sql, query_result, numeric_values, step_query}, ...]
    - state["rag_docs"]: list of retrieved documents (from rag_tool)
    - state["api_results"]: list of API call records:
        { "tool": <name>, "input": {...}, "output": <value or dict> }

    It does NOT try to infer numbers with regex; it trusts tool outputs.
    """

    # Conversation / trace context
    memory_view = build_memory_view(state)

    # ---- Build a simple, structured view of tool outputs ----

    # SQL block - show all SQL query results
    sql_block = ""
    sql_results = state.get("sql_results") or []
    if sql_results:
        lines = ["SQL_RESULTS:"]
        for idx, sql_res in enumerate(sql_results, 1):
            generated_sql = sql_res.get("generated_sql")
            query_result = sql_res.get("query_result")
            numeric_values = sql_res.get("numeric_values", [])
            step_query = sql_res.get("step_query", "")
            
            lines.append(f"  [Query {idx}] {step_query[:60]}...")
            if generated_sql:
                lines.append(f"    generated_sql: {generated_sql}")
            if query_result is not None:
                lines.append(f"    query_result: {query_result}")
            if numeric_values:
                lines.append(f"    numeric_values: {numeric_values}")
        sql_block = "\n".join(lines)

    # RAG block
    rag_block = ""
    rag_docs = state.get("rag_docs")
    if rag_docs:
        # Expecting a list of dicts with "content" (based on your rag_tool)
        lines = ["RAG_RESULTS:"]
        for i, doc in enumerate(rag_docs[:3]):  # show at most 3 docs
            content = doc.get("content") if isinstance(doc, dict) else str(doc)
            if content:
                trimmed = content[:600]  # keep it readable
                lines.append(f"  [Doc {i+1}] {trimmed}")
        rag_block = "\n".join(lines)

    # API block
    api_block = ""
    api_results = state.get("api_results") or []
    if api_results:
        lines = ["API_RESULTS:"]
        for i, call in enumerate(api_results):
            tool_name = call.get("tool")
            inputs = call.get("input")
            output = call.get("output")
            lines.append(f"  [Call {i+1}] tool: {tool_name}")
            if inputs is not None:
                lines.append(f"    input: {inputs}")
            if output is not None:
                lines.append(f"    output: {output}")
        api_block = "\n".join(lines)

    # Combine tool context
    tool_context_parts = [b for b in [sql_block, rag_block, api_block] if b]
    tool_context = "\n\n".join(tool_context_parts) if tool_context_parts else "No tools were called."

    # ---- System prompt: how to use these tool outputs ----

    system_prompt = """You are a helpful assistant that answers user questions based on tool execution results.

You are given:
- A user query.
- A structured summary of TOOL CONTEXT (SQL, RAG, API results).
- A short conversation/trajectory history.

Your job:
- Synthesize a clear, direct answer to the user query.
- Base ALL numeric values and factual details on the TOOL CONTEXT.
- Do NOT invent or approximate numbers. If a value is not present, say you don't have it.

Guidelines:

1. SQL results (SQL_RESULTS)
   - Treat values from SQL as the ground truth for data stored in the database (e.g., totals, counts).
   - There may be multiple SQL queries executed. Each query's results are shown separately.
   - Use numeric_values from each query result to get the actual numeric data.

2. RAG results (RAG_RESULTS)
   - These contain policy / knowledge text (e.g., currency policy, pricing policy, company info).
   - Use them to:
     - Get currency conversion rates.
     - Get descriptive information (e.g., "digital music retailer", "founded in 2009", etc.).
   - Do NOT assume policies that are not mentioned in these documents.

3. API results (API_RESULTS)
   - These are concrete TOOL CALL outputs (e.g., currency conversion results).
   - When reporting converted amounts, COPY the values from API outputs exactly.
   - Do NOT recompute or change these numbers yourself.
   - If multiple API calls are present, choose the ones that are relevant to the user query (e.g., correct currency).

4. Answering:
   - Always answer the user's question fully and succinctly.
   - If the question involves both USD and a converted currency:
       - State the base USD amount from SQL.
       - State the converted amount from the API output.
       - If a conversion rate is clearly present in RAG text or API inputs, mention it explicitly.
   - If any information is missing or ambiguous in the tool context, say so instead of guessing.

Return ONLY the final answer text for the user (no JSON, no markdown fences)."""

    # ---- User prompt ----

    user_prompt = f"""User query:
{state["user_query"]}

TOOL CONTEXT:
{tool_context}

Conversation / execution history (summary):
{memory_view}

Write the final answer to the user, following the system instructions above."""

    try:
        # Choose model
        config = state.get("config", {})
        model_name = (
            config.get("answer_model")
            or config.get("model_name")
            or "gpt-4o-mini"
        )
        llm, actual_model_name = init_llm(model_name)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        # Call LLM with tracing
        response = await traced_llm_call(
            node_name="answer",
            state=state,
            llm_callable=llm,
            llm_input=messages,
            model_name=actual_model_name
        )

        state["answer_draft"] = response.content.strip()
        state["ready_for_reflection"] = True

    except Exception as e:
        state["answer_draft"] = f"I encountered an error while generating the answer: {str(e)}"
        state["ready_for_reflection"] = True

    # IMPORTANT: mark episode as done so the controller routes to "end"
    # This prevents infinite loops when answer step is reached
    state["done"] = True

    return state
