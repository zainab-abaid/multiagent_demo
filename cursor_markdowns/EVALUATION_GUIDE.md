# Agent Evaluation Guide

## Overview

This guide explains how to evaluate the multi-agent system on composite queries using the deterministic evaluation framework in `evaluate_agent.py`.

## Evaluation System Architecture

The evaluation system uses a **three-layer deterministic approach**:

### 1. Component Extraction (LLM)
- **Purpose**: Extract structured values from the agent's free-form answer
- **Input**: Question, agent answer, expected answer keys
- **Output**: Dictionary with extracted values (or `null` if missing)
- **Why**: Converts natural language answers into structured data for comparison

### 2. Deterministic Scoring (Python)
- **Purpose**: Compare extracted values to ground truth
- **Metrics**:
  - **Correctness**: Fraction of components with correct values
  - **Completeness**: Fraction of components present
  - **Accuracy**: Fraction of numeric components correct (with tolerance)
  - **Final Answer Correct**: All components present and correct
- **Why**: Fully deterministic, reproducible scoring tied to ground truth

### 3. Step-wise Trajectory Analysis (Python)
- **Purpose**: Analyze each step of the pipeline separately
- **Checks**:
  - **SQL Query Correctness**: Compare generated SQL to ground truth SQL
  - **SQL Result Correctness**: Compare SQL result value to expected
  - **API Correctness**: Validate currency conversion calculations
  - **Tool Usage**: Track which tools were used
  - **Propagation Correctness**: Check if tool outputs correctly propagated to final answer
- **Why**: Identifies exactly where errors occur in the pipeline

## Dataset: Composite Queries

The `composite_queries.jsonl` file contains 10 composite queries that require multiple tools. Each entry includes:

- `id`: Unique identifier (comp_1 through comp_10)
- `question`: The query text
- `requires_tools`: List of tools needed (`sql_tool`, `rag_tool`, `api_tool`)
- `base_query_id`: Original query ID from `chinook_tasks_extended.jsonl`
- `expected_answer`: Verified correct answer with all components
- `ground_truth_sql`: Correct SQL query for the database portion

## Expected Answers and Ground Truth SQL

### comp_1: Revenue 2011 → EUR
- **Question**: "What was the total revenue for 2011, and what would that amount be if converted to EUR based on the store's currency policy?"
- **Expected Answer**:
  - `revenue_usd`: 469.58
  - `revenue_eur`: 554.10
  - `conversion_rate`: "1 USD = 1.18 EUR"
- **Ground Truth SQL**: `SELECT ROUND(SUM(Total), 2) FROM Invoice WHERE strftime('%Y', InvoiceDate) = '2011'`
- **Tools**: sql_tool, rag_tool, api_tool

### comp_2: Comedy Tracks → GBP Value
- **Question**: "How many tracks are in the Comedy genre, and what would be the estimated total value in GBP if each track costs the average price according to the store's pricing policy?"
- **Expected Answer**:
  - `track_count`: 17
  - `estimated_value_usd`: 17.85
  - `estimated_value_gbp`: 24.45
  - `average_price_per_track`: 1.05
  - `conversion_rate`: "1 USD = 1.37 GBP"
- **Ground Truth SQL**: `SELECT COUNT(*) FROM Track t JOIN Genre g ON t.GenreId = g.GenreId WHERE g.Name = 'Comedy'`
- **Tools**: sql_tool, rag_tool, api_tool

### comp_3: Revenue 2013 → CAD
- **Question**: "What is the total revenue for 2013, and how much would that be in CAD based on the store's currency conversion rates?"
- **Expected Answer**:
  - `revenue_usd`: 450.58
  - `revenue_cad`: 563.23
  - `conversion_rate`: "1 USD = 1.25 CAD"
- **Ground Truth SQL**: `SELECT ROUND(SUM(Total), 2) FROM Invoice WHERE strftime('%Y', InvoiceDate) = '2013'`
- **Tools**: sql_tool, rag_tool, api_tool

### comp_4: Bruce Dickinson Albums + Store Founded
- **Question**: "How many albums does Bruce Dickinson have, and when was the store founded according to company information?"
- **Expected Answer**:
  - `album_count`: 1
  - `store_founded`: 2009
- **Ground Truth SQL**: `SELECT COUNT(*) FROM Album a JOIN Artist ar ON a.ArtistId = ar.ArtistId WHERE ar.Name = 'Bruce Dickinson'`
- **Tools**: sql_tool, rag_tool

### comp_5: Average Invoice → JPY
- **Question**: "What is the average invoice total, and what would that average be in JPY according to the store's currency policy?"
- **Expected Answer**:
  - `average_invoice_usd`: 5.65
  - `average_invoice_jpy`: 621.71
  - `conversion_rate`: "1 USD = 110 JPY"
- **Ground Truth SQL**: `SELECT ROUND(AVG(Total), 2) FROM Invoice`
- **Tools**: sql_tool, rag_tool, api_tool

### comp_6: Invoices 2009 + Price Range
- **Question**: "How many invoices were issued in 2009, and what is the typical price range for individual tracks according to the store's pricing policy?"
- **Expected Answer**:
  - `invoice_count`: 83
  - `price_range`: "$0.99 - $1.99 USD per track"
- **Ground Truth SQL**: `SELECT COUNT(*) FROM Invoice WHERE strftime('%Y', InvoiceDate) = '2009'`
- **Tools**: sql_tool, rag_tool

### comp_7: Highest Invoice → AUD
- **Question**: "What is the highest invoice total, and what would that amount be in AUD based on the store's currency conversion rates?"
- **Expected Answer**:
  - `highest_invoice_usd`: 25.86
  - `highest_invoice_aud`: 34.91
  - `conversion_rate`: "1 USD = 1.35 AUD"
- **Ground Truth SQL**: `SELECT MAX(Total) FROM Invoice`
- **Tools**: sql_tool, rag_tool, api_tool

### comp_8: Genre Count + Popularity Info
- **Question**: "How many different genres are there, and what information does the store have about genre popularity according to the documents?"
- **Expected Answer**:
  - `genre_count`: 25
  - `rag_info`: "Store offers tracks across 25 different genres. Rock and Pop typically have highest sales volumes. Latin music has seen significant growth, especially in 2013."
- **Ground Truth SQL**: `SELECT COUNT(*) FROM Genre`
- **Tools**: sql_tool, rag_tool

### comp_9: Total Invoices → GBP Revenue
- **Question**: "What is the total number of invoices, and what would the total revenue be in GBP if we assume the average invoice amount and convert using the store's currency policy?"
- **Expected Answer**:
  - `invoice_count`: 412
  - `estimated_revenue_usd`: 2328.60
  - `estimated_revenue_gbp`: 3190.18
  - `average_invoice`: 5.65
  - `conversion_rate`: "1 USD = 1.37 GBP"
- **Ground Truth SQL**: `SELECT COUNT(*) FROM Invoice`
- **Tools**: sql_tool, rag_tool, api_tool

### comp_10: Customer Count + Business Model
- **Question**: "How many customers are there, and what is the store's business model according to company information?"
- **Expected Answer**:
  - `customer_count`: 59
  - `business_model`: "Digital Music Retailer, Per-track and per-album sales, Global customer base"
- **Ground Truth SQL**: `SELECT COUNT(*) FROM Customer`
- **Tools**: sql_tool, rag_tool

## Running Evaluation

### Basic Usage (Default Dataset)
```bash
python evaluate_agent.py
```

This runs evaluation on all queries in `composite_queries.jsonl`.

### Custom Dataset
```bash
python evaluate_agent.py path/to/your/queries.jsonl
```

### Interactive Testing (Single Query)
```bash
python debug_agent.py "What was the total revenue for 2011, and what would that amount be if converted to EUR based on the store's currency policy?"
```

## Output Structure

Each evaluation run creates a session directory:

```
logs/session/eval_<timestamp>/
├── query1/
│   ├── trajectory.jsonl      # Full agent execution trace with all events
│   ├── query_metadata.json   # Query info, expected answer, ground truth SQL
│   └── evaluation.json       # Complete evaluation results
├── query2/
│   └── ...
├── ...
└── summary.json              # Overall evaluation summary with statistics
```

## Terminal Output

The script prints detailed evaluation results for each query:

```
[1/10] Query comp_1
Question: What was the total revenue for 2011...

Agent Answer:
The total revenue for 2011 was $469.58. When converted to EUR...

Judge Evaluation:
  Overall Score: 0.85/1.00
  Correctness: 0.90/1.00
  Completeness: 1.00/1.00
  Accuracy: 0.80/1.00
  Final Answer Correct: False

  Step Analysis:
    SQL Query Correct: True
      Similarity: 0.95
    SQL Result Correct: True
    SQL Value: 469.58
    Generated SQL: SELECT ROUND(SUM(Total), 2) FROM Invoice...
    API Step Correct: True
    API Value: 554.10
    Tools Used: {'sql_tool': True, 'rag_tool': True, 'api_tool': True}
    Propagation Correct: {'revenue_usd': True, 'revenue_eur': False}

  Extracted Values:
    revenue_usd: 469.58
    revenue_eur: 399.14
    conversion_rate: null
```

## Evaluation Metrics Explained

### Component-Level Metrics

- **Correctness**: Fraction of expected components that have correct values
  - Numeric: Within tolerance (default 0.01)
  - String: Case-insensitive substring match
- **Completeness**: Fraction of expected components that are present (not null)
- **Accuracy**: Fraction of numeric components that are correct
- **Final Answer Correct**: Boolean - all components present AND correct

### Step-Level Metrics

- **SQL Query Correct**: Whether generated SQL matches ground truth (normalized comparison)
- **SQL Result Correct**: Whether SQL result value matches expected
- **API Step Correct**: Whether currency conversion calculation is correct
- **Propagation Correct**: Whether tool outputs correctly propagated to final answer

### Overall Score

Weighted combination:
```
overall_score = 0.4 × correctness + 0.4 × accuracy + 0.2 × completeness
```

## Understanding Evaluation Results

### SQL Query vs SQL Result

The system tracks two separate SQL metrics:

1. **SQL Query Correctness**: Is the SQL query itself correct?
   - Compares generated SQL to ground truth SQL
   - Normalized comparison (handles formatting differences)
   - Shows similarity score and differences

2. **SQL Result Correctness**: Does the SQL query return the correct value?
   - Compares SQL result to expected numeric value
   - Useful when SQL is syntactically different but semantically equivalent

### Error Diagnosis

The evaluation helps identify where errors occur:

- **SQL Query Wrong**: Problem in SQL generation → Check planner/SQL tool
- **SQL Query Correct, Result Wrong**: Problem in SQL execution or data → Check database
- **SQL Correct, API Wrong**: Problem in currency conversion → Check API tool
- **SQL & API Correct, Answer Wrong**: Problem in answer synthesis → Check answer node
- **Propagation Issues**: Tool outputs not correctly used in final answer

## Extending the Dataset

To evaluate on a different dataset:

1. Create a JSONL file with the same structure as `composite_queries.jsonl`
2. Each line should be a JSON object with:
   ```json
   {
     "id": "unique_id",
     "question": "Your question here",
     "requires_tools": ["sql_tool", "rag_tool", "api_tool"],
     "base_query_id": 1000,
     "notes": "Optional notes",
     "expected_answer": {
       "component1": value1,
       "component2": value2
     },
     "ground_truth_sql": "SELECT ... FROM ... WHERE ..."
   }
   ```
3. Run: `python evaluate_agent.py your_dataset.jsonl`

## Configuration

### Environment Variables

Set LLM models for different components in `.env`:

```bash
# Judge LLM (used for component extraction)
JUDGE_MODEL=gpt-4o-mini

# Agent LLMs (used during execution)
PLANNER_MODEL=gpt-4o-mini
SQL_MODEL=gpt-4o-mini
ANSWER_MODEL=gpt-4o-mini
```

### Numeric Tolerance

Default tolerance for numeric comparisons is `1e-2` (0.01). This can be adjusted in the `score_components` function.

## Notes

- **Currency Conversion Rates**: From `documents/pricing_policy.txt`
- **Estimation Queries**: Some queries (comp_2, comp_9) require estimation based on average prices
- **RAG Information**: May vary slightly in wording but should contain key facts
- **SQL Queries**: Should use correct aggregation (SUM, COUNT, AVG, MAX) as appropriate
- **Ground Truth SQL**: Used for query-level correctness, not just result correctness

## Troubleshooting

### Evaluation Fails
- Check that `.env` has `OPENAI_API_KEY` set
- Verify `composite_queries.jsonl` is valid JSONL
- Ensure database file `Chinook.db` exists

### SQL Comparison Issues
- SQL normalization handles formatting differences
- Check `sql_comparison.differences` for specific issues
- Similarity score < 0.8 usually indicates significant differences

### Missing Components
- Check `extracted` values in evaluation output
- Component extractor may miss values if answer format is unexpected
- Verify expected answer keys match what the agent should produce
