# Multi-Agent System with SQL, RAG, and API Tools

A LangGraph-based multi-agent system for answering questions using SQL queries, RAG retrieval, and API calls.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment:**
   Create a `.env` file with:
   ```
   OPENAI_API_KEY=your_key_here
   ```

3. **Initialize RAG store:**
   ```bash
   python init_rag_store.py
   ```

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
  - `graph.py` - LangGraph workflow definition
  - `nodes_*.py` - Node implementations (planner, tools, answer)
  - `tools_*.py` - Tool implementations (SQL, RAG, API)
  - `state.py` - Agent state definitions
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
- **Comprehensive Evaluation**: Tests against ground truth with detailed scoring

## Notes

- The SQL database (`Chinook.db`) is automatically downloaded if missing
- The RAG vector store (`chroma_db/`) is generated when you run `init_rag_store.py`
- Evaluation logs are saved to `logs/session/` directory

