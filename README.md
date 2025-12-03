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
   MAX_REPLANS=5  # Maximum number of replanning attempts
   RAG_TOP_K=5  # Number of documents to retrieve in RAG queries
   RAG_CHUNK_SIZE=512  # Chunk size for document splitting in RAG indexing
   ```

3. **Set up RAG documents:**
   - Place your documents in a `documents/` folder at the repository root
   - Then initialize the RAG vector store:
   ```bash
   python init_rag_store.py
   ```
   Note: This must be run before using the agent. The vector store will be saved to `chroma_db/` and reused in future runs.

4. **Run the agent:**
   ```bash
   python debug_agent.py "What is the total revenue for 2011?"
   ```

5. **Run evaluation:**
   ```bash
   python evaluate_agent.py composite_queries.jsonl
   # Or test a specific query:
   python evaluate_agent.py composite_queries.jsonl comp_9
   ```

## Project Structure

- `agent/` - Agent implementation with nodes and tools
  - `graph.py` - LangGraph workflow definition with replanning loop
  - `nodes_*.py` - Node implementations (planner, tools, answer, controller)
  - `tools_*.py` - Tool implementations (SQL, RAG, API)
  - `state.py` - Agent state definitions (includes replanning state)
  - `llm_utils.py` - LLM initialization supporting OpenAI and Groq
  - `tracing.py` - Tracing utilities for logging LLM calls and tool execution
- `debug_agent.py` - Interactive debugging tool
- `evaluate_agent.py` - Evaluation framework with ground truth comparison
- `composite_queries.jsonl` - Ground truth evaluation data
- `documents/` - RAG documents (company info, pricing policy, genre info)
- `init_rag_store.py` - Script to initialize RAG vector store

## Features

- **SQL Tool**: Natural language to SQL query generation for Chinook database
- **RAG Tool**: Document retrieval for company policies and information
- **API Tool**: Currency conversion and other API operations
- **Multi-step Planning**: Breaks down complex queries into tool execution plans
- **Replanning Loop**: Automatically re-evaluates execution results and generates corrective steps if needed
- **Multi-Provider LLM Support**: Supports both OpenAI and Groq models (use `groq/<model>` prefix for Groq)
- **Comprehensive Evaluation**: Tests against ground truth with detailed scoring

## Agent Flow

The agent follows this execution flow:

1. **Input Node**: Initializes state from user query
2. **Planner Node**: Creates initial execution plan with tool calls
3. **Plan Controller**: Routes to appropriate next step
4. **Tool Caller**: Executes tools (SQL, RAG, API) based on plan
5. **Replanning Check**: After all tools complete, planner re-evaluates:
   - Checks if enough information was gathered
   - Identifies any errors or missing data
   - Generates new corrective steps if needed (up to `MAX_REPLANS` times)
6. **Answer Node**: Synthesizes final answer from tool results
7. **Reflection Node**: Analyzes execution for improvements

The replanning loop ensures the agent can recover from errors and gather missing information before generating the final answer.

## Notes

- The SQL database (`Chinook.db`) is automatically downloaded if missing
- **RAG Setup**: Place documents in `documents/` folder at repo root, then run `init_rag_store.py` to create the vector store (`chroma_db/`)
- Evaluation logs are saved to `logs/session/` directory

