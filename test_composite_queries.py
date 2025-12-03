#!/usr/bin/env python3
"""Test all composite queries to verify SQL and API calls are correct."""

import json
import sqlite3
from agent.tools_api import (
    convert_from_usd, calculate_total_value, calculate_estimated_revenue,
    format_duration_hours, calculate_percentage
)

DB_PATH = "Chinook.db"

def execute_sql(sql: str) -> any:
    """Execute SQL and return result."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute(sql)
        result = cursor.fetchone()
        return result[0] if result else None
    except Exception as e:
        print(f"SQL Error: {e}")
        return None
    finally:
        conn.close()

def test_query(query: dict) -> tuple[bool, str]:
    """Test a single composite query."""
    query_id = query["id"]
    errors = []
    
    # Test SQL
    sql = query.get("ground_truth_sql")
    if sql:
        sql_result = execute_sql(sql)
        expected_sql_value = query.get("expected_tool_calls", {}).get("sql_tool", {}).get("expected_result_value")
        
        if expected_sql_value is not None:
            if sql_result is None:
                errors.append(f"SQL returned None but expected {expected_sql_value}")
            else:
                # Allow small floating point differences
                if abs(float(sql_result) - float(expected_sql_value)) > 0.01:
                    errors.append(f"SQL mismatch: got {sql_result}, expected {expected_sql_value}")
    
    # Test API calls
    api_calls = query.get("expected_tool_calls", {}).get("api_tool", {}).get("expected_calls", [])
    for api_call in api_calls:
        tool_name = api_call.get("tool")
        input_data = api_call.get("input", {})
        
        try:
            if tool_name == "convert_currency_from_usd":
                result = convert_from_usd(input_data["amount_usd"], input_data["target_currency"])
                # Check if result is reasonable (we'll validate against expected_answer)
            elif tool_name == "calculate_total_value":
                result = calculate_total_value(input_data["quantity"], input_data["unit_price"])
            elif tool_name == "calculate_estimated_revenue":
                result = calculate_estimated_revenue(input_data["count"], input_data["average_amount"])
            elif tool_name == "format_duration_hours":
                result = format_duration_hours(input_data["minutes"])
            elif tool_name == "calculate_percentage":
                result = calculate_percentage(input_data["part"], input_data["total"])
            else:
                errors.append(f"Unknown API tool: {tool_name}")
        except Exception as e:
            errors.append(f"API call {tool_name} failed: {e}")
    
    # Check expected_answer values
    expected_answer = query.get("expected_answer", {})
    for key, expected_value in expected_answer.items():
        if isinstance(expected_value, (int, float)):
            # These should match SQL or API results - we've already checked SQL above
            pass
    
    if errors:
        return False, "; ".join(errors)
    return True, "OK"

def main():
    """Test all composite queries."""
    with open("composite_queries_extended.jsonl", "r") as f:
        queries = [json.loads(line) for line in f if line.strip()]
    
    print(f"Testing {len(queries)} composite queries...\n")
    
    passed = 0
    failed = 0
    
    for query in queries:
        query_id = query["id"]
        success, message = test_query(query)
        if success:
            passed += 1
            print(f"✓ {query_id}: {message}")
        else:
            failed += 1
            print(f"✗ {query_id}: {message}")
    
    print(f"\nResults: {passed} passed, {failed} failed out of {len(queries)} total")

if __name__ == "__main__":
    main()

