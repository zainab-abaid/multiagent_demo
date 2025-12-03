# Composite Queries Dataset

This dataset contains queries that require multiple tools (SQL + RAG + API) to answer completely.

## Dataset File

`composite_queries.jsonl` - Contains 10 composite queries

## Query Structure

Each query in the dataset:
- Requires **at least 2 tools** (SQL + RAG, or SQL + RAG + API)
- Is based on a simple query from `chinook_tasks_extended.jsonl`
- Extends the simple query with additional requirements

## Example Queries

### Query 1: Revenue + Currency Conversion
**Question**: "What was the total revenue for 2011, and what would that amount be if converted to EUR based on the store's currency policy?"

**Required Tools**:
- SQL: Get total revenue for 2011
- RAG: Get EUR conversion rate from pricing policy
- API: Convert USD to EUR

### Query 2: Count + Pricing + Currency
**Question**: "How many tracks are in the Comedy genre, and what would be the estimated total value in GBP if each track costs the average price according to the store's pricing policy?"

**Required Tools**:
- SQL: Count Comedy tracks
- RAG: Get average track price from pricing policy
- API: Convert USD to GBP

### Query 3: Revenue + Currency (CAD)
**Question**: "What is the total revenue for 2013, and how much would that be in CAD based on the store's currency conversion rates?"

**Required Tools**:
- SQL: Get total revenue for 2013
- RAG: Get CAD conversion rate
- API: Convert USD to CAD

## Usage

Run any query with:
```bash
python debug_agent.py "YOUR_QUERY_HERE"
```

Or use the queries from the JSONL file:
```bash
python -c "
import json
with open('composite_queries.jsonl') as f:
    for line in f:
        q = json.loads(line)
        print(q['question'])
" | head -1 | xargs python debug_agent.py
```

## Notes

- All queries require SQL tool for database access
- Most queries require RAG tool for policy/company information
- Currency conversion queries require API tool
- Some queries combine SQL + RAG only (no currency conversion needed)

