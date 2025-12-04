# Multi-Agent System with SQL, RAG, and API Tools

A LangGraph-based multi-agent system for answering questions using SQL queries, RAG retrieval, and API calls. Features replanning capabilities and support for both OpenAI and Groq models.

## Prerequisites

**IMPORTANT: You must initialize the RAG vector store before using the agent.**

1. Place your documents in a `documents/` folder at the repository root
2. Run the initialization script:
   ```bash
   # With uv
   uv run python init_rag_store.py
   
   # With pip
   python init_rag_store.py
   ```
   This creates the vector store in `chroma_db/` and must be completed before running the agent.

## Quick Start

### Option 1: Using `uv` (Recommended)

1. **Install `uv`** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   Or via pip: `pip install uv`

2. **Set up environment variables:**
   Create a `.env` file with your API keys and configuration (see Option 2 below for details).

3. **Install dependencies and run scripts:**
   ```bash
   # Install dependencies
   uv sync
   
   # Run scripts with uv
   uv run python init_rag_store.py
   uv run python debug_agent.py "What is the total revenue for 2011?"
   uv run python simplified_evaluate_agent.py composite_queries_extended.jsonl
   ```

### Option 2: Using `pip`

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables:**
   Create a `.env` file with your API keys and configuration:
   ```
   # Required API keys
   OPENAI_API_KEY=your_key_here
   GROQ_API_KEY=your_groq_key_here  # Optional, for Groq models
   
   # Model configuration (all optional, defaults shown)
   PLANNER_MODEL=gpt-4o-mini  # or groq/llama-3.1-70b-versatile
   SQL_MODEL=gpt-4o-mini
   ANSWER_MODEL=gpt-4o-mini
   REFLECTION_MODEL=gpt-4o-mini
   JUDGE_MODEL=gpt-4o-mini
   
   # Agent behavior configuration
   MAX_REPLANS=5  # Maximum number of replanning attempts (default: 3)
   RAG_TOP_K=5  # Number of documents to retrieve in RAG queries (default: 5)
   RAG_CHUNK_SIZE=512  # Chunk size for document splitting in RAG indexing (default: 512)
   ```
   
   All configuration options are set via environment variables in the `.env` file.

3. **Run the agent:**
   
   **Note**: If using `uv`, prefix commands with `uv run` (e.g., `uv run python debug_agent.py ...`)
   
   **For single query debugging and detailed tracing:**
   ```bash
   # With uv
   uv run python debug_agent.py "What is the total revenue for 2011?"
   
   # With pip
   python debug_agent.py "What is the total revenue for 2011?"
   ```
   Use `debug_agent.py` when you want to:
   - Test any single query and see a simple log.
   
   **For simplified evaluation (answer matching only):**
   ```bash
   # Run all queries from the default file
   uv run python simplified_evaluate_agent.py composite_queries_extended.jsonl
   # or: python simplified_evaluate_agent.py composite_queries_extended.jsonl
   
   # Run a specific query by ID
   uv run python simplified_evaluate_agent.py composite_queries_extended.jsonl comp_3300
   
   # Run a random subset of queries
   uv run python simplified_evaluate_agent.py composite_queries_extended.jsonl --random 10
   ```
   Use `simplified_evaluate_agent.py` when you want to:
   - Quickly evaluate multiple queries against groundtruth (in composite_queries_extended.jsonl)
   - See answer matching scores from an LLM judge
   - Get a simple average score across queries
   - Focus on final answer quality
   - Note: Some improvements in the evaluation and groundtruth are pending, as the groundtruth is AI generated. We have verified it to the best of our ability, but occasional errors persist.

   For testing on any query that is in the provided dataset, we reocmmend you run simplified_evaluate_agent.py.

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
- `debug_agent.py` - Single-query debugging tool with tracing
- `simplified_evaluate_agent.py` - Simplified evaluation (answer matching only)
- `composite_queries_extended.jsonl` - Ground truth evaluation data 
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
- **Evaluation Tools**: 
  - `debug_agent.py`: Single-query debugging with detailed tracing
  - `simplified_evaluate_agent.py`: Quick evaluation focusing on answer matching only
  - Supports random sampling with `--random N` option

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
- **RAG Setup (REQUIRED)**: 
  - Place documents in `documents/` folder at repo root
  - Run `python init_rag_store.py` to create the vector store (`chroma_db/`)
  - **This is a prerequisite** - the agent will not work without this step
- **Recursion Limit**: Set to 150 to support multi-step workflows with replanning (configured when invoking the graph)
- **State Management**: 
  - Tool results are stored in structured format: `sql_results`, `rag_results`, `api_results`
  - Memory view provides structured summaries of tool executions for the planner
- **Evaluation**: 
  - Logs are saved to `logs/session/eval_<timestamp>/` directory for simplified_evaluate_agent.py and as a single debug_* file if running debug_agent.py.
  - Each query has its own subdirectory with `trajectory.jsonl`, `evaluation.json`, and `final_state.json`

