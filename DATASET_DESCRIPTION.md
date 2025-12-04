# Evaluation Dataset Description

This document describes the evaluation dataset used for testing the multi-agent system.

## Dataset File

`composite_queries_extended.jsonl` - Contains 50 composite queries that require multiple tools to answer completely.

## Query Structure

Each query in the dataset:
- Requires **at least 2 tools** (SQL + RAG, SQL + API, or SQL + RAG + API)
- Is based on a simple query from `chinook_tasks_extended.jsonl` (referenced by `base_query_id`)
- Extends the simple query with additional requirements (e.g., currency conversion, artist information, time formatting)

## Query Fields

Each query entry contains:
- `id`: Unique identifier (e.g., "comp_3300")
- `question`: The natural language query
- `requires_tools`: List of required tools (e.g., ["sql_tool", "rag_tool", "api_tool"])
- `base_query_id`: Reference to the original simple query
- `notes`: Brief description of query type
- `expected_answer`: Structured expected answer with component keys
- `ground_truth_sql`: The SQL query that should be generated
- `expected_tool_calls`: Expected tool call details for evaluation

## Tool Usage Statistics

- **SQL Tool**: Used in all 50 queries (100%)
- **RAG Tool**: Used in 42 queries (84%)
- **API Tool**: Used in 40 queries (80%)

## Query Types

### 1. SQL + RAG (Artist/Genre Information)
**Example**: "What is the average number of tracks per album for Body Count? What information does the store have about this artist?"

**Required Tools**:
- SQL: Calculate average tracks per album
- RAG: Retrieve artist information from documents

### 2. SQL + RAG + API (Currency Conversion)
**Example**: "What is the total quantity of Drama tracks sold in 2012? What would be the estimated total value in EUR if each track costs the average price according to the store's pricing policy?"

**Required Tools**:
- SQL: Get quantity of tracks sold
- RAG: Get average price and currency conversion rate from pricing policy
- API: Calculate total value and convert currency

### 3. SQL + API (Time Formatting)
**Example**: "What is the total length in minutes of all tracks in albums by Black Sabbath? How many hours and minutes is that?"

**Required Tools**:
- SQL: Calculate total duration in minutes
- API: Format duration as hours and minutes

### 4. SQL + RAG (Null Results)
**Example**: "What is the total quantity of Pop tracks sold in 2013? What information does the store have about this genre?"

**Required Tools**:
- SQL: Query returns `null` (no data for that year/genre combination)
- RAG: Retrieve genre information

## Expected Answer Structure

The `expected_answer` field contains structured components that the agent should extract:

- **Numeric values**: `quantity`, `count`, `average`, `revenue_usd`, `minutes`, `hours`, etc.
- **Currency conversions**: `estimated_value_usd`, `estimated_value_eur`, `estimated_value_gbp`, `conversion_rate`
- **Time formatting**: `formatted` (e.g., "1 hours and 21.61 minutes")
- **RAG information**: `rag_info` (descriptive text about artists, genres, composers)
- **Null handling**: `result: null` when SQL queries return no data

## Usage

Run evaluation on the full dataset:
```bash
python simplified_evaluate_agent.py composite_queries_extended.jsonl
```

Run a specific query:
```bash
python simplified_evaluate_agent.py composite_queries_extended.jsonl comp_3300
```

Run a random subset:
```bash
python simplified_evaluate_agent.py composite_queries_extended.jsonl --random 10
```

Test a single query interactively:
```bash
python debug_agent.py "What is the average number of tracks per album for Body Count? What information does the store have about this artist?"
```

## Notes

- All queries require SQL tool for database access
- Most queries require RAG tool for policy/company/artist information
- Currency conversion queries require API tool
- Some queries test null result handling (SQL returns `null`)
- The ground truth data is AI-generated and has been verified to the best of our ability, but occasional errors may persist
