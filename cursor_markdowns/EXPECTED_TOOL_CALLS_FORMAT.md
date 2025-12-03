# Expected Tool Calls Format

This document describes the format for `expected_tool_calls` in the ground truth data for composite queries.

## Format

```json
{
  "expected_tool_calls": {
    "sql_tool": {
      "expected_result_value": 469.58
    },
    "rag_tool": {
      "expected_content": "Currency conversion rate: 1 USD = 1.18 EUR"
    },
    "api_tool": {
      "expected_calls": [
        {
          "tool": "convert_currency_from_usd",
          "input": {
            "amount_usd": 469.58,
            "target_currency": "EUR"
          },
          "output": 554.10
        }
      ]
    }
  }
}
```

## Fields

### sql_tool
- `expected_result_value` (number, optional): The expected numeric result from the SQL query. This is compared against `state.sql_result.usd_value`.

### rag_tool
- `expected_content` (string, optional): A description of the expected information that should be found in the retrieved documents. Used for semantic matching with LLM.

### api_tool
- `expected_calls` (array, optional): List of expected API tool calls. Each call should specify:
  - `tool` (string): The tool name from the API registry (e.g., "convert_currency_from_usd")
  - `input` (object): The expected input parameters
  - `output` (number/string, optional): The expected output (not always needed for correctness check)

## Example for comp_1

```json
{
  "id": "comp_1",
  "question": "What was the total revenue for 2011, and what would that amount be if converted to EUR based on the store's currency policy?",
  "expected_answer": {
    "revenue_usd": 469.58,
    "revenue_eur": 554.10,
    "conversion_rate": "1 USD = 1.18 EUR"
  },
  "ground_truth_sql": "SELECT ROUND(SUM(Total), 2) FROM Invoice WHERE strftime('%Y', InvoiceDate) = '2011'",
  "expected_tool_calls": {
    "sql_tool": {
      "expected_result_value": 469.58
    },
    "rag_tool": {
      "expected_content": "Currency conversion rate: 1 USD = 1.18 EUR"
    },
    "api_tool": {
      "expected_calls": [
        {
          "tool": "convert_currency_from_usd",
          "input": {
            "amount_usd": 469.58,
            "target_currency": "EUR"
          }
        }
      ]
    }
  }
}
```

## Notes

- If a tool is not expected, simply omit it from `expected_tool_calls`
- The evaluation will only check tools that are specified in `expected_tool_calls`
- For RAG, use semantic matching (LLM-based) rather than exact string matching
- For API tools, the evaluation checks that the tool name and input parameters match (output is optional)

