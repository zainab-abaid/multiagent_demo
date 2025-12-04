# Multi-Agent System with SQL, RAG, and API Tools

A LangGraph-based multi-agent system for answering questions using SQL queries, RAG retrieval, and API calls. Features replanning capabilities and support for both OpenAI and Groq models.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment:**
   Create a `.env` file with:
   ```
   OPENAI_API_KEY=your_key_here
   GROQ_API_KEY=your_groq_key_here  # Optional, for Groq models
   ```
   
   You can also configure model-specific settings:
   ```
   PLANNER_MODEL=gpt-4o-mini  # or groq/llama-3.1-70b-versatile
   SQL_MODEL=gpt-4o-mini
   ANSWER_MODEL=gpt-4o-mini
   REFLECTION_MODEL=gpt-4o-mini
   JUDGE_MODEL=gpt-4o-mini
   MAX_REPLANS=5  # Maximum number of replanning attempts (default: 3)
   RAG_TOP_K=5  # Number of documents to retrieve in RAG queries (default: 5)
   RAG_CHUNK_SIZE=512  # Chunk size for document splitting in RAG indexing (default: 512)
   ```

3. **Set up RAG documents:**
   - Place your documents in a `documents/` folder at the repository root
   - Then initialize the RAG vector store:
   ```bash
   python init_rag_store.py
   ```
   Note: This must be run before using the agent. The vector store will be saved to `chroma_db/` and reused in future runs.

4. **Run the agent (single query debugging):**
   ```bash
   python debug_agent.py "What is the total revenue for 2011?"
   ```

5. **Run evaluation:**
   ```bash
   # Run all queries from the default file
   python evaluate_agent.py composite_queries_extended.jsonl
   
   # Run a specific query by ID
   python evaluate_agent.py composite_queries_extended.jsonl comp_3300
   
   # Run a random subset of queries
   python evaluate_agent.py composite_queries_extended.jsonl --random 10
   ```

## Project Structure

- `agent/` - Agent implementation with nodes and tools
  - `graph.py` - LangGraph workflow definition with replanning loop
  - `nodes_*.py` - Node implementations:
    - `nodes_input.py` - Initializes state from user query
    - `nodes_planner.py` - Creates initial plans and handles replanning
    - `nodes_controller.py` - Advances cursor and increments replan_count (state mutations)
    - `nodes_tools.py` - Executes SQL, RAG, and API tools
    - `nodes_answer.py` - Generates final answer from tool results
    - `nodes_reflection.py` - Analyzes execution for improvements
  - `tools_*.py` - Tool implementations:
    - `tools_sql.py` - Natural language to SQL conversion and execution
    - `tools_rag.py` - Document retrieval using LlamaIndex and ChromaDB
    - `tools_api.py` - API functions (currency conversion, calculations)
    - `tools_api_registry.py` - API tool registry with schemas
    - `api_router.py` - LLM-based router for selecting API tools
  - `state.py` - Agent state definitions (includes `replan_count`, `rag_results`, `sql_results`, `api_results`)
  - `llm_utils.py` - LLM initialization supporting OpenAI and Groq
  - `tracing.py` - Tracing utilities for logging LLM calls and tool execution
- `debug_agent.py` - Single-query debugging tool with detailed tracing
- `evaluate_agent.py` - Evaluation framework with ground truth comparison
- `composite_queries_extended.jsonl` - Ground truth evaluation data (default)
- `composite_queries.jsonl` - Alternative evaluation dataset
- `documents/` - RAG documents (company info, pricing policy, genre info, artist info)
- `init_rag_store.py` - Script to initialize RAG vector store (must run before using agent)

## Features

- **SQL Tool**: Natural language to SQL query generation for Chinook database
  - Automatically includes database schema in prompts
  - Handles markdown code block extraction from LLM responses
  - Returns structured results with query, SQL, and execution status
- **RAG Tool**: Document retrieval for company policies and information
  - Uses LlamaIndex with ChromaDB for persistent vector storage
  - Configurable `RAG_TOP_K` and `RAG_CHUNK_SIZE` via environment variables
  - Tracks queries and results in `rag_results` for memory view
- **API Tool**: Currency conversion and other API operations
  - LLM-based router selects appropriate API functions
  - Supports currency conversion (with rates from RAG), calculations, formatting
  - Functions: `convert_currency_from_usd`, `convert_currency_to_usd`, `calculate_total_value`, `format_duration_hours`, `calculate_percentage`
- **Multi-step Planning**: Breaks down complex queries into structured execution plans
  - Planner generates specific queries for each tool step (not just full user query)
  - Distinguishes between DATA (SQL) and KNOWLEDGE (RAG) queries
- **Replanning Loop**: Automatically re-evaluates execution results
  - Checks if queries were already tried (prevents infinite loops)
  - Generates corrective steps if needed (up to `MAX_REPLANS` times)
  - Tracks replanning attempts in state and memory view
- **Multi-Provider LLM Support**: Supports both OpenAI and Groq models (use `groq/<model>` prefix for Groq)
- **Comprehensive Evaluation**: Tests against ground truth with detailed scoring
  - Supports random sampling with `--random N` option
  - Detailed trajectory logging for debugging

## Agent Flow

The agent follows this execution flow:

1. **Input Node**: Initializes state from user query
2. **Planner Node**: Creates initial execution plan with structured tool calls
3. **Plan Controller Node**: 
   - Advances step cursor for "think" steps and unknown action types
   - Increments `replan_count` when about to replan (ensures state mutations persist)
4. **Route Controller**: Makes routing decisions based on plan state:
   - Routes to `tool_caller` if there are more tool_call steps
   - Routes to `planner` for replanning if all steps done but not ready to answer (up to `MAX_REPLANS` times)
   - Routes to `answer` if ready to generate final answer
   - Routes to `end` if episode is complete
5. **Tool Caller Node**: Executes tools (SQL, RAG, API) based on plan steps:
   - **SQL Tool**: Converts natural language to SQL and executes against Chinook database
   - **RAG Tool**: Retrieves relevant documents from vector store using specific queries
   - **API Tool**: Routes to appropriate API functions (currency conversion, calculations, formatting)
6. **Replanning**: After tool execution, if not ready to answer:
   - Planner analyzes execution history (structured memory view with tool results)
   - Checks if queries were already tried (prevents infinite loops)
   - Generates new corrective steps if needed (respects `MAX_REPLANS` limit)
   - Replanning count is tracked and displayed in memory view
7. **Answer Node**: Synthesizes final answer from all tool results
8. **Reflection Node**: Analyzes execution for improvements

The replanning loop ensures the agent can recover from errors and gather missing information before generating the final answer. The recursion limit is set to 150 to support multi-step workflows with replanning.

## Notes

- **SQL Database**: The Chinook database (`Chinook.db`) is automatically downloaded if missing
- **RAG Setup**: 
  - Place documents in `documents/` folder at repo root
  - Run `python init_rag_store.py` to create the vector store (`chroma_db/`)
  - This must be done before using the agent
- **Recursion Limit**: Set to 150 to support multi-step workflows with replanning (configured when invoking the graph)
- **State Management**: 
  - `replan_count` is incremented in `plan_controller_node` (a proper node) to ensure state mutations persist
  - Tool results are stored in structured format: `sql_results`, `rag_results`, `api_results`
  - Memory view provides structured summaries of tool executions for the planner
- **Evaluation**: 
  - Logs are saved to `logs/session/eval_<timestamp>/` directory
  - Each query has its own subdirectory with `trajectory.jsonl`, `evaluation.json`, and `final_state.json`

