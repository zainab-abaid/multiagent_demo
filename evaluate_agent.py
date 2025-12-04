#!/usr/bin/env python
"""Evaluate agent on composite queries dataset with simplified evaluation."""

import json
import sys
import time
import asyncio
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, SystemMessage
from agent.llm_utils import init_llm

from agent.graph import build_agent_graph
from agent.state import create_initial_state, AgentConfig

load_dotenv()


# ============================================================================
# Constants: LLM Prompts
# ============================================================================

PROMPT_EXTRACT_COMPONENTS = """You are a precise information extraction assistant.

Extract values from the agent's answer for each key in the expected answer.

For each key:
- Find the corresponding value in the agent's answer
- Extract ALL relevant information that matches the expected component
- For string values with multiple facts (like "rag_info" or "business_model"), extract ALL facts mentioned, even if phrased differently
- Match semantic meaning, not exact wording - if the agent says something equivalent, extract it
- For fields like "rag_info" that contain multiple statements, extract all related information the agent mentioned
- Return numbers as numbers, strings as strings
- Return null only if the component is truly missing or ambiguous

Return ONLY a JSON object with the SAME KEYS as the expected answer.
Make sure to extract complete information, especially for descriptive fields like "business_model" or "rag_info" that contain multiple facts."""

PROMPT_SEMANTIC_MATCH = """You are a semantic correctness judge.

Determine if two text values are semantically equivalent.
Consider them equivalent if they convey the same core information, even with different wording.

Return ONLY a JSON object:
{
  "semantically_correct": true or false
}"""

PROMPT_RAG_CONTENT = """You are a RAG content correctness judge.

Your task is to compare expected content with retrieved documents and count how many individual facts from the expected content are present.

Steps:
1. Break down the expected content into individual, distinct facts (e.g., "Digital Music Retailer", "Per-track and per-album sales", "Global customer base" = 3 facts)
2. Count how many of these facts are present in the retrieved documents
3. Return the count as a fraction

Return ONLY a JSON object:
{
  "matched_facts": <number of facts present>,
  "total_facts": <total number of facts in expected content>
}

Be specific - each distinct piece of information counts as one fact."""

PROMPT_PLAN_COMPARISON = """You are an evaluator comparing an expected plan with an actual plan.
Your task is to count how many of the expected steps are present in the actual plan.

A step is considered "present" if:
1. The action_type matches (tool_call, think, answer)
2. For tool_call steps, the tool name matches (sql_tool, rag_tool, api_tool)
3. The step serves a similar purpose (semantic similarity in description)

IMPORTANT: 
- Ignore extra fields in the actual plan (like "query" field for sql_tool steps) - they are just implementation details
- Focus on matching the tool type and the semantic meaning of the description
- Minor wording differences in descriptions are acceptable

Return a JSON object with:
{
  "matched_count": <number of expected steps that are present>,
  "total_expected": <total number of expected steps>
}

Be lenient - if the tools match and the descriptions are semantically similar, count it as a match."""


# ============================================================================
# Utilities: File I/O and State Management
# ============================================================================



def append_trajectory_events(
    state: Dict[str, Any], 
    output_file: Path, 
    saved_event_count: int
) -> int:
    """
    Append new trajectory events to the JSONL file incrementally.
    
    Returns:
        int: New count of saved events
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    trajectory = state.get("trajectory", [])
    new_events = trajectory[saved_event_count:]
    
    if new_events:
        # Append mode - open in append mode to add new events
        with open(output_file, "a") as f:
            for event in new_events:
                f.write(json.dumps(event) + "\n")
                f.flush()  # Force write to disk immediately
    
    return len(trajectory)


def _prepare_expected_tool_calls(query_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract expected_tool_calls and add expected_plan if present."""
    expected_tool_calls = query_data.get("expected_tool_calls", {}).copy()
    if "expected_plan" in query_data:
        expected_tool_calls["expected_plan"] = query_data["expected_plan"]
    return expected_tool_calls


def _invoke_llm(llm, messages: List) -> Any:
    """Invoke LLM synchronously or asynchronously."""
    if hasattr(llm, 'ainvoke'):
        return llm.ainvoke(messages)
    return llm.invoke(messages)


async def _parse_llm_json_response(response: Any) -> Dict[str, Any]:
    """Parse JSON from LLM response, handling markdown code blocks."""
    text = response.content.strip()
    
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
        text = text.replace("```json", "").replace("```", "").strip()
    
    return json.loads(text)


# ============================================================================
# Answer Evaluation: Component Extraction and Scoring
# ============================================================================

async def extract_answer_components(
    question: str,
    agent_answer: str,
    expected_answer: Dict[str, Any],
    judge_llm
) -> Dict[str, Any]:
    """Extract structured values from agent's answer based on expected_answer keys."""
    user_prompt = f"""Question: {question}

Agent's Answer:
{agent_answer}

Expected Answer Keys and their expected values (for reference):
{json.dumps(expected_answer, indent=2)}

Extract the values the agent claimed for EACH of these keys.
- For each key, extract ALL relevant information from the agent's answer that matches what's expected
- If a key expects multiple parts (like "Digital Music Retailer, Per-track and per-album sales, Global customer base"), extract ALL parts that the agent mentioned
- Be thorough - look through the entire answer, not just the first mention
- For string values, combine all relevant phrases the agent used

Special handling:
- For "conversion_rate": If the agent mentions a conversion (e.g., "converted to EUR", "in CAD"), extract the rate in the format "1 [FROM] = [RATE] [TO]" or "[RATE] [TO]" if you can infer it from the conversion amounts mentioned
- For numeric fields that are close (within tolerance), extract them even if slightly different

Return ONLY a JSON object with exactly these keys."""

    try:
        messages = [SystemMessage(content=PROMPT_EXTRACT_COMPONENTS), HumanMessage(content=user_prompt)]
        response = await _invoke_llm(judge_llm, messages)
        extracted = await _parse_llm_json_response(response)
        
        # Ensure all expected keys are present
        return {key: extracted.get(key, None) for key in expected_answer.keys()}
    except Exception:
        return {key: None for key in expected_answer.keys()}


async def judge_string_match(
    question: str,
    expected_value: str,
    extracted_value: str,
    component_key: str,
    judge_llm
) -> bool:
    """Use LLM to judge if two strings are semantically equivalent."""
    user_prompt = f"""Question: {question}

Component: {component_key}

Expected: {expected_value}

Extracted: {extracted_value}

Are these semantically equivalent? Return JSON with semantically_correct (true/false)."""

    try:
        messages = [SystemMessage(content=PROMPT_SEMANTIC_MATCH), HumanMessage(content=user_prompt)]
        response = await _invoke_llm(judge_llm, messages)
        judgment = await _parse_llm_json_response(response)
        return judgment.get("semantically_correct", False)
    except Exception:
        return False


def _check_conversion_rate_implied(expected: Dict[str, Any], extracted: Dict[str, Any], numeric_tolerance: float) -> bool:
    """Check if conversion_rate can be skipped because amounts are correct."""
    usd_keys = [k for k in expected.keys() if "usd" in k.lower() and k != "conversion_rate"]
    converted_keys = [k for k in expected.keys() if any(c in k.lower() for c in ["eur", "gbp", "jpy", "cad", "aud"]) and k != "conversion_rate"]
    
    if not (usd_keys and converted_keys):
        return False
    
    # Check if USD amount is correct
    usd_correct = any(
        isinstance(expected[k], (int, float)) and extracted.get(k) is not None
        for k in usd_keys
        if _numeric_match(expected[k], extracted.get(k), numeric_tolerance)
    )
    
    # Check if converted amount is correct
    converted_correct = any(
        isinstance(expected[k], (int, float)) and extracted.get(k) is not None
        for k in converted_keys
        if _numeric_match(expected[k], extracted.get(k), numeric_tolerance)
    )
    
    return usd_correct and converted_correct


def _extract_numeric_from_sql_result(query_result: str) -> Optional[float]:
    """Extract numeric value from SQL query result string."""
    if not query_result:
        return None
    try:
        return float(query_result.strip())
    except (ValueError, AttributeError, TypeError):
        return None


def _numeric_match(expected: float, actual: Any, base_tolerance: float) -> bool:
    """Check if two numbers match within tolerance."""
    try:
        got_num = float(actual)
        exp_num = float(expected)
        tolerance = max(base_tolerance, abs(exp_num) * 0.02) if exp_num > 10 else base_tolerance
        return abs(got_num - exp_num) <= tolerance
    except Exception:
        return False


async def score_answer_correctness(
    expected: Dict[str, Any],
    extracted: Dict[str, Any],
    question: str,
    agent_answer: str,
    judge_llm,
    numeric_tolerance: float = 2.0
) -> Dict[str, Any]:
    """Score answer correctness: fraction of components that are correct."""
    if not expected:
        return {"score": 1.0, "component_feedback": []}
    
    correct = 0
    total = len(expected)
    component_feedback = []
    conversion_rate_skipped = False
    
    for key in expected:
        exp_val = expected[key]
        got_val = extracted.get(key, None)
        
        # Special handling: skip conversion_rate if amounts are correct
        if key == "conversion_rate" and got_val is None:
            if _check_conversion_rate_implied(expected, extracted, numeric_tolerance):
                conversion_rate_skipped = True
                component_feedback.append({
                    "component": key,
                    "status": "skipped",
                    "reason": "Conversion rate implied by correct conversion amounts"
                })
            continue
        
        if got_val is None:
            component_feedback.append({
                "component": key,
                "status": "missing",
                "expected": str(exp_val)[:100],
                "reason": "Component not found in extracted answer"
            })
            continue
        
        # Numeric comparison
        if isinstance(exp_val, (int, float)):
            if _numeric_match(exp_val, got_val, numeric_tolerance):
                correct += 1
                component_feedback.append({
                    "component": key,
                    "status": "correct",
                    "expected": str(exp_val),
                    "extracted": str(got_val)
                })
            else:
                got_num = float(got_val)
                exp_num = float(exp_val)
                diff = abs(got_num - exp_num)
                tolerance = max(numeric_tolerance, abs(exp_num) * 0.02) if exp_num > 10 else numeric_tolerance
                component_feedback.append({
                    "component": key,
                    "status": "incorrect",
                    "expected": str(exp_val),
                    "extracted": str(got_val),
                    "reason": f"Numeric mismatch: difference of {diff:.2f} exceeds tolerance of {tolerance:.2f}"
                })
        else:
            # String comparison - use LLM for semantic matching
            is_correct = await judge_string_match(question, str(exp_val), str(got_val), key, judge_llm)
            if is_correct:
                correct += 1
                component_feedback.append({
                    "component": key,
                    "status": "correct",
                    "expected": str(exp_val)[:150],
                    "extracted": str(got_val)[:150]
                })
            else:
                component_feedback.append({
                    "component": key,
                    "status": "incorrect",
                    "expected": str(exp_val)[:150],
                    "extracted": str(got_val)[:150] if got_val else "null",
                    "reason": "Semantic mismatch: extracted value does not match expected meaning"
                })
    
    if conversion_rate_skipped:
        total -= 1
    
    score = correct / total if total > 0 else 0.0
    return {
        "score": score,
        "component_feedback": component_feedback,
        "correct_count": correct,
        "total_count": total
    }


# ============================================================================
# Tool Correctness Checks: SQL, RAG, API
# ============================================================================

def check_sql_tool(
    sql_result: Optional[Dict[str, Any]],
    sql_results: Optional[List[Dict[str, Any]]],
    expected_tool_calls: Dict[str, Any],
    ground_truth_sql: Optional[str] = None,
    check_query: bool = False
) -> Dict[str, Any]:
    """Check SQL tool correctness by comparing results and optionally queries."""
    expected_sql = expected_tool_calls.get("sql_tool", {})
    expected_value = expected_sql.get("expected_result_value")
    expected_queries = expected_sql.get("expected_sql_queries", [])
    expected_results = expected_sql.get("expected_results", [])
    
    result = {
        "correct": False,
        "query_correct": None,
        "generated_sql": None,
        "result_value": None,
        "queries_matched": None,
        "queries_total": None,
    }
    
    all_sql_results = sql_results if sql_results else ([sql_result] if sql_result else [])
    if not all_sql_results:
        return result
    
    # Normalize ground truth SQL if provided
    if ground_truth_sql and not expected_queries:
        expected_queries = [q.strip() for q in ground_truth_sql.split(";") if q.strip()]
    
    has_multiple_queries = len(expected_queries) > 1 or (ground_truth_sql and ";" in ground_truth_sql)
    
    if has_multiple_queries and expected_queries:
        return _check_multiple_sql_queries(all_sql_results, expected_queries, expected_results, expected_value, check_query, result)
    else:
        return _check_single_sql_query(all_sql_results[0] if all_sql_results else sql_result, expected_value, ground_truth_sql, check_query, result)


def _check_multiple_sql_queries(
    all_sql_results: List[Dict[str, Any]],
    expected_queries: List[str],
    expected_results: List[float],
    expected_value: Optional[float],
    check_query: bool,
    result: Dict[str, Any]
) -> Dict[str, Any]:
    """Check correctness when multiple SQL queries are expected."""
    result["queries_total"] = len(expected_queries)
    
    generated_queries = [r.get("generated_sql") for r in all_sql_results if r and r.get("generated_sql")]
    # Extract numeric values from query_result strings
    result_values = []
    for r in all_sql_results:
        if r and r.get("query_result"):
            query_result = r.get("query_result")
            # Try to extract numeric value from query result string
            numeric_val = _extract_numeric_from_sql_result(query_result)
            if numeric_val is not None:
                result_values.append(numeric_val)
    
    # Check query matching if requested
    if check_query:
        matched = 0
        for exp_query in expected_queries:
            exp_norm = " ".join(exp_query.upper().split())
            for gen_query in generated_queries:
                gen_norm = " ".join(gen_query.upper().split())
                if gen_norm == exp_norm:
                    matched += 1
                    break
        result["queries_matched"] = matched
        result["query_correct"] = matched == len(expected_queries)
    
    # Check result values
    if expected_results and len(expected_results) == len(result_values):
        result["correct"] = all(
            result_values[i] is not None and _numeric_match(expected_results[i], result_values[i], 2.0)
            for i in range(len(expected_results))
        )
    elif expected_value is not None and result_values:
        result["correct"] = _numeric_match(expected_value, result_values[0], 2.0)
    
    return result


def _check_single_sql_query(
    sql_res: Optional[Dict[str, Any]],
    expected_value: Optional[float],
    ground_truth_sql: Optional[str],
    check_query: bool,
    result: Dict[str, Any]
) -> Dict[str, Any]:
    """Check correctness for a single SQL query."""
    if not sql_res:
        return result
    
    generated_sql = sql_res.get("generated_sql")
    query_result = sql_res.get("query_result")
    result_value = _extract_numeric_from_sql_result(query_result) if query_result else None
    
    result["generated_sql"] = generated_sql
    result["result_value"] = result_value
    
    # Check result value
    if expected_value is not None and result_value is not None:
        result["correct"] = _numeric_match(expected_value, result_value, 2.0)
    
    # Check query syntax if requested
    if check_query and ground_truth_sql and generated_sql:
        truth_queries = [q.strip() for q in ground_truth_sql.split(";") if q.strip()]
        gen_norm = " ".join(generated_sql.upper().split())
        if len(truth_queries) == 1:
            truth_norm = " ".join(truth_queries[0].upper().split())
            result["query_correct"] = gen_norm == truth_norm
    
    return result


async def check_rag_tool(
    rag_docs: Optional[List],
    expected_tool_calls: Dict[str, Any],
    judge_llm
) -> Dict[str, Any]:
    """Check RAG tool correctness by counting how many expected facts are present."""
    expected_rag = expected_tool_calls.get("rag_tool", {})
    expected_content = expected_rag.get("expected_content")
    
    result = {
        "correct": False,
        "score": 0.0,
        "fraction": "0/0",
        "matched_facts": 0,
        "total_facts": 0,
        "retrieved_docs": None,
    }
    
    if not rag_docs or not expected_content:
        return result
    
    # Extract content from RAG docs
    doc_contents = [doc.get("content", "") for doc in rag_docs if isinstance(doc, dict) and doc.get("content")]
    result["retrieved_docs"] = len(doc_contents)
    
    if not doc_contents:
        return result
    
    # Use LLM to count how many expected facts are present
    combined_docs = "\n\n".join(doc_contents[:3])
    user_prompt = f"""Expected content to find:
{expected_content}

Retrieved documents:
{combined_docs[:2000]}

Break down the expected content into individual facts, then count how many of those facts are present in the retrieved documents. Return JSON."""

    try:
        messages = [SystemMessage(content=PROMPT_RAG_CONTENT), HumanMessage(content=user_prompt)]
        response = await _invoke_llm(judge_llm, messages)
        judgment = await _parse_llm_json_response(response)
        
        matched = judgment.get("matched_facts", 0)
        total = judgment.get("total_facts", 1)
        
        result["matched_facts"] = matched
        result["total_facts"] = total
        result["fraction"] = f"{matched}/{total}"
        result["score"] = matched / total if total > 0 else 0.0
        result["correct"] = (matched == total) and (total > 0)
    except Exception:
        pass
    
    return result


def _match_api_call_inputs(exp_input: Dict[str, Any], act_input: Dict[str, Any]) -> tuple[bool, List[str]]:
    """Check if API call inputs match, returning (matches, issues)."""
    param_issues = []
    
    for key, exp_val in exp_input.items():
        act_val = act_input.get(key)
        
        if isinstance(exp_val, (int, float)):
            tolerance = max(2.0, abs(exp_val) * 0.02) if abs(exp_val) > 10 else 2.0
            if act_val is None:
                param_issues.append(f"{key}: missing (expected {exp_val})")
                return False, param_issues
            if not _numeric_match(exp_val, act_val, 2.0):
                diff = abs(float(act_val) - float(exp_val))
                param_issues.append(f"{key}: got {act_val} (expected {exp_val}, diff={diff:.2f}, tolerance={tolerance:.2f})")
                return False, param_issues
        elif act_val != exp_val:
            param_issues.append(f"{key}: got {act_val} (expected {exp_val})")
            return False, param_issues
    
    return True, []


def check_api_tool(
    api_results: Optional[List],
    expected_tool_calls: Dict[str, Any]
) -> Dict[str, Any]:
    """Check API tool correctness by comparing tool name, inputs, and expected output."""
    expected_api = expected_tool_calls.get("api_tool", {})
    expected_calls = expected_api.get("expected_calls", [])
    
    result = {
        "correct": False,
        "calls_made": [],
        "calls_expected": expected_calls,
        "match_details": [],
        "mismatch_details": [],
        "matches": 0,
        "total_expected": len(expected_calls),
    }
    
    if not api_results:
        result["correct"] = len(expected_calls) == 0
        return result
    
    # Extract actual calls
    actual_calls = [
        {"tool": call.get("tool"), "input": call.get("input"), "output": call.get("output")}
        for call in api_results
        if isinstance(call, dict)
    ]
    
    result["calls_made"] = actual_calls
    
    if not expected_calls:
        result["correct"] = len(actual_calls) == 0
        return result
    
    if len(actual_calls) != len(expected_calls):
        return result
    
    # Check each expected call has a matching actual call
    matches = 0
    for exp_call in expected_calls:
        exp_tool = exp_call.get("tool")
        exp_input = exp_call.get("input", {})
        
        matched = False
        for act_call in actual_calls:
            if act_call.get("tool") == exp_tool:
                act_input = act_call.get("input", {})
                input_match, param_issues = _match_api_call_inputs(exp_input, act_input)
                
                if input_match:
                    matches += 1
                    matched = True
                    result["match_details"].append(f"  ✅ {exp_tool}: inputs match")
                    break
                else:
                    result["mismatch_details"].append({
                        "tool": exp_tool,
                        "expected_input": exp_input,
                        "actual_input": act_input,
                        "issues": param_issues
                    })
        
        if not matched:
            result["mismatch_details"].append({
                "tool": exp_tool,
                "expected_input": exp_input,
                "actual_input": None,
                "issues": ["No matching tool call found"]
            })
    
    result["correct"] = matches == len(expected_calls)
    result["matches"] = matches
    return result


# ============================================================================
# Plan Correctness Check
# ============================================================================

def _fallback_plan_matching(expected_steps: List[Dict], actual_steps: List[Dict]) -> tuple[int, int]:
    """Fallback plan matching when LLM judge fails - matches tools in order."""
    expected_tools = []
    for step in expected_steps:
        if step.get("action_type") == "tool_call":
            tool = step.get("tool")
            if tool:
                expected_tools.append(tool)
        elif step.get("action_type") == "answer":
            expected_tools.append("answer")
    
    actual_tools = []
    for step in actual_steps:
        if step.get("action_type") == "tool_call":
            tool = step.get("tool")
            if tool:
                actual_tools.append(tool)
        elif step.get("action_type") == "answer":
            actual_tools.append("answer")
    
    # Match tools in order (allows extra steps in actual plan)
    matched = 0
    actual_idx = 0
    for expected_tool in expected_tools:
        found = False
        while actual_idx < len(actual_tools):
            if actual_tools[actual_idx] == expected_tool:
                matched += 1
                found = True
                actual_idx += 1
                break
            actual_idx += 1
        if not found:
                break
        
    return matched, len(expected_tools)


async def check_plan_correctness(
    plan: Optional[Dict[str, Any]],
    expected_plan: Optional[Dict[str, Any]],
    judge_llm
) -> Dict[str, Any]:
    """Check plan correctness using LLM to compare expected steps with actual steps."""
    result = {
        "score": 0.0,
        "fraction": "0/0",
        "matched_steps": 0,
        "total_expected_steps": 0,
    }
    
    if not plan or "steps" not in plan:
        return result
    
    if not expected_plan or "steps" not in expected_plan:
        return result
    
    expected_steps = expected_plan.get("steps", [])
    actual_steps = plan.get("steps", [])
    
    result["total_expected_steps"] = len(expected_steps)
    
    if len(expected_steps) == 0:
        return result
    
    # Try LLM-based matching first
    try:
        user_prompt = f"""Expected plan steps:
{json.dumps(expected_steps, indent=2)}

Actual plan steps:
{json.dumps(actual_steps, indent=2)}

Count how many expected steps are present in the actual plan. Return the JSON."""
        
        messages = [SystemMessage(content=PROMPT_PLAN_COMPARISON), HumanMessage(content=user_prompt)]
        response = await judge_llm.ainvoke(messages)
        judgment = await _parse_llm_json_response(response)
        
        matched_count = judgment.get("matched_count", 0)
        total_expected = judgment.get("total_expected", len(expected_steps))
        
        result["matched_steps"] = matched_count
        result["total_expected_steps"] = total_expected
        result["fraction"] = f"{matched_count}/{total_expected}"
        result["score"] = matched_count / total_expected if total_expected > 0 else 0.0
    except Exception:
        # Fallback to simple tool-based matching
        matched, total = _fallback_plan_matching(expected_steps, actual_steps)
        result["matched_steps"] = matched
        result["total_expected_steps"] = total
        result["fraction"] = f"{matched}/{total}"
        result["score"] = matched / total if total > 0 else 0.0
    
    return result


# ============================================================================
# Main Evaluation Orchestrator
# ============================================================================

async def evaluate_simplified(
    question: str,
    agent_answer: str,
    expected_answer: Dict[str, Any],
    final_state: Dict[str, Any],
    expected_tool_calls: Dict[str, Any],
    ground_truth_sql: Optional[str] = None,
    judge_llm = None
) -> Dict[str, Any]:
    """Main evaluation function - orchestrates all correctness checks."""
    # 1. Extract answer components
    extracted = await extract_answer_components(question, agent_answer, expected_answer, judge_llm)
    
    # 2. Score answer correctness
    answer_score_result = await score_answer_correctness(
        expected_answer, extracted, question, agent_answer, judge_llm
    )
    answer_match_score = answer_score_result["score"]
    component_feedback = answer_score_result.get("component_feedback", [])
    
    # 3. Check plan correctness (use initial plan if available)
    plan_history = final_state.get("plan_history", [])
    initial_plan = plan_history[0] if plan_history else final_state.get("plan")
    
    expected_plan = expected_tool_calls.get("expected_plan")
    plan_check = await check_plan_correctness(initial_plan, expected_plan, judge_llm)
    
    # 4. Check tool calls
    sql_results = final_state.get("sql_results")
    rag_docs = final_state.get("rag_docs")
    api_results = final_state.get("api_results")
    
    sql_check = _check_sql_tool_correctness(expected_tool_calls, sql_results, ground_truth_sql)
    rag_check = await _check_rag_tool_correctness(expected_tool_calls, rag_docs, judge_llm)
    api_check = _check_api_tool_correctness(expected_tool_calls, api_results)
    
    # 5. Compute tool call correctness score
    tool_call_correctness = _compute_tool_call_correctness(sql_check, rag_check, api_check)
    
    # 6. Report replanning stats
    replan_count = final_state.get("replan_count", 0)
    
    # 7. Calculate overall_score as average of all component scores
    overall_score = (answer_match_score + plan_check["score"] + tool_call_correctness) / 3.0
    
    return {
        "overall_score": overall_score,
        "answer_match_score": answer_match_score,
        "plan_correctness": plan_check["score"],
        "tool_call_correctness": tool_call_correctness,
        "replan_count": replan_count,
        "extracted": extracted,
        "plan_check": plan_check,
        "component_feedback": component_feedback,
        "answer_score_details": answer_score_result,
        "tool_checks": {
            "sql": sql_check,
            "rag": rag_check,
            "api": api_check,
        },
    }


def _check_sql_tool_correctness(
    expected_tool_calls: Dict[str, Any],
    sql_results: Optional[List],
    ground_truth_sql: Optional[str]
) -> Dict[str, Any]:
    """Check SQL tool correctness if expected."""
    sql_check = {"correct": None, "query_correct": None}
    
    if not expected_tool_calls.get("sql_tool"):
        return sql_check
    
    sql_result = None
    if sql_results:
        sql_result = sql_results[-1] if sql_results else None
    
    sql_check = check_sql_tool(
        sql_result,
        sql_results,
        expected_tool_calls,
        ground_truth_sql=ground_truth_sql,
        check_query=False
    )
    
    # Only check query syntax if SQL result values are wrong
    if sql_check.get("correct") is False:
        query_check = check_sql_tool(
            sql_result,
            sql_results,
            expected_tool_calls,
            ground_truth_sql=ground_truth_sql,
            check_query=True
        )
        sql_check["query_correct"] = query_check.get("query_correct")
        sql_check["queries_matched"] = query_check.get("queries_matched")
        sql_check["queries_total"] = query_check.get("queries_total")
    
    return sql_check


async def _check_rag_tool_correctness(
    expected_tool_calls: Dict[str, Any],
    rag_docs: Optional[List],
    judge_llm
) -> Dict[str, Any]:
    """Check RAG tool correctness if expected."""
    rag_check = {"correct": None}
    if expected_tool_calls.get("rag_tool"):
        rag_check = await check_rag_tool(rag_docs, expected_tool_calls, judge_llm)
    return rag_check


def _check_api_tool_correctness(
    expected_tool_calls: Dict[str, Any],
    api_results: Optional[List]
) -> Dict[str, Any]:
    """Check API tool correctness if expected."""
    api_check = {"correct": None}
    if expected_tool_calls.get("api_tool"):
        api_check = check_api_tool(api_results, expected_tool_calls)
    return api_check


def _compute_tool_call_correctness(
    sql_check: Dict[str, Any],
    rag_check: Dict[str, Any],
    api_check: Dict[str, Any]
) -> float:
    """Compute overall tool call correctness score from individual tool checks."""
    tool_scores = []
    
    if sql_check.get("correct") is not None:
        tool_scores.append(1.0 if sql_check["correct"] else 0.0)
    
    if rag_check.get("correct") is not None:
        rag_score = rag_check.get("score", 1.0 if rag_check["correct"] else 0.0)
        tool_scores.append(rag_score)
    
    if api_check.get("correct") is not None:
        tool_scores.append(1.0 if api_check["correct"] else 0.0)
    
    return sum(tool_scores) / len(tool_scores) if tool_scores else 1.0


# ============================================================================
# Query Execution
# ============================================================================

async def run_query(
    query_data: Dict[str, Any],
    query_num: int,
    total_queries: int,
    graph,
    session_dir: Path,
    judge_llm
) -> Dict[str, Any]:
    """Run a single query through the agent and evaluate with simplified judge."""
    query_id = query_data["id"]
    question = query_data["question"]
    expected_answer = query_data.get("expected_answer", {})
    
    print(f"\n[{query_num}/{total_queries}] Query {query_id}")
    print(f"Question: {question}")
    
    # Create log directory for this query
    query_log_dir = session_dir / f"query{query_num}"
    query_log_dir.mkdir(parents=True, exist_ok=True)
    
    # Save query metadata
    with open(query_log_dir / "query_metadata.json", "w") as f:
        json.dump({
            "query_id": query_id,
            "question": question,
            "expected_answer": expected_answer,
            "requires_tools": query_data.get("requires_tools", []),
            "expected_tool_calls": query_data.get("expected_tool_calls", {}),
            "ground_truth_sql": query_data.get("ground_truth_sql"),
        }, f, indent=2)
    
    # Run the agent
    config = AgentConfig()
    initial_state = create_initial_state(question, config)
    
    # Initialize trajectory file (create/truncate)
    trajectory_file = query_log_dir / "trajectory.jsonl"
    trajectory_file.parent.mkdir(parents=True, exist_ok=True)
    trajectory_file.write_text("")  # Clear file at start
    
    final_state = None
    saved_event_count = 0  # Track how many events we've already saved
    
    try:
        async for step in graph.astream(initial_state, stream_mode="values", config={"recursion_limit": 150}):
            final_state = step
            # Save trajectory incrementally after each step
            saved_event_count = append_trajectory_events(
                step, 
                trajectory_file, 
                saved_event_count
            )
    except Exception as e:
        print(f"  [ERROR] {e}")
        # Even on error, try to save whatever state we have
        if final_state is None:
            final_state = initial_state
        # Save final state even on error
        with open(query_log_dir / "final_state.json", "w") as f:
            json.dump(final_state, f, indent=2, default=str)
        return {
            "query_id": query_id,
            "question": question,
            "error": str(e),
            "agent_answer": None,
            "expected_answer": expected_answer,
            "judge_evaluation": None,
        }
    
    if final_state is None:
        final_state = initial_state
    
    # Save final state (trajectory already saved incrementally)
    with open(query_log_dir / "final_state.json", "w") as f:
        json.dump(final_state, f, indent=2, default=str)
    
    # Get agent's answer
    agent_answer = final_state.get("answer_draft", "No answer generated.")
    print(f"\nAgent Answer:\n{agent_answer}")
    
    # Prepare expected_tool_calls and evaluate
    expected_tool_calls = _prepare_expected_tool_calls(query_data)
    judge_evaluation = await evaluate_simplified(
        question=question,
        agent_answer=agent_answer,
        expected_answer=expected_answer,
        final_state=final_state,
        expected_tool_calls=expected_tool_calls,
        ground_truth_sql=query_data.get("ground_truth_sql"),
        judge_llm=judge_llm
    )
    
    # Save evaluation result
    with open(query_log_dir / "evaluation.json", "w") as f:
        json.dump({
            "query_id": query_id,
            "question": question,
            "agent_answer": agent_answer,
            "expected_answer": expected_answer,
            "judge_evaluation": judge_evaluation,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)
    
    return {
        "query_id": query_id,
        "question": question,
        "agent_answer": agent_answer,
        "expected_answer": expected_answer,
        "judge_evaluation": judge_evaluation,
    }


# ============================================================================
# Output Formatting
# ============================================================================

def _print_evaluation_summary(eval_data: Dict[str, Any]) -> None:
    """Print formatted evaluation summary for a single query."""
    overall_score = eval_data.get('overall_score', 0)
    answer_match_score = eval_data.get('answer_match_score', 0)
    
    print(f"\nEvaluation:")
    print(f"  Overall Score: {overall_score:.2f}/1.00 (average of all components)")
    print(f"    Answer Match Score: {answer_match_score:.2f}/1.00")
    
    # Component feedback
    component_feedback = eval_data.get("component_feedback", [])
    if overall_score < 1.0 and component_feedback:
        print(f"  Overall Score Breakdown:")
        answer_score_details = eval_data.get("answer_score_details", {})
        correct_count = answer_score_details.get("correct_count", 0)
        total_count = answer_score_details.get("total_count", 0)
        print(f"    Components: {correct_count}/{total_count} correct")
        for feedback in component_feedback:
            _print_component_feedback(feedback)
    
    # Plan correctness
    plan_check = eval_data.get("plan_check", {})
    if plan_check:
        plan_score = eval_data.get("plan_correctness", 0.0)
        plan_fraction = plan_check.get("fraction", "0/0")
        print(f"  Plan Correctness: {plan_score:.2f}/1.00 ({plan_fraction})")
    
    # Tool call correctness
    print(f"  Tool Call Correctness: {eval_data.get('tool_call_correctness', 0):.2f}/1.00")
    
    # Replan count
    replan_count = eval_data.get("replan_count", 0)
    print(f"  Replan Count: {replan_count}")
    
    # Individual tool checks
    tool_checks = eval_data.get("tool_checks", {})
    if tool_checks:
        print(f"  Tool Checks:")
        _print_tool_checks(tool_checks)


def _print_component_feedback(feedback: Dict[str, Any]) -> None:
    """Print feedback for a single component."""
    status = feedback.get("status", "unknown")
    component = feedback.get("component", "unknown")
    
    if status == "correct":
        print(f"      ✅ {component}")
    elif status == "incorrect":
        reason = feedback.get("reason", "Mismatch")
        expected = feedback.get("expected", "N/A")
        extracted = feedback.get("extracted", "N/A")
        print(f"      ❌ {component}: {reason}")
        print(f"         Expected: {expected[:80]}...")
        print(f"         Extracted: {extracted[:80]}...")
    elif status == "missing":
        expected = feedback.get("expected", "N/A")
        print(f"      ❌ {component}: Missing from answer")
        print(f"         Expected: {expected[:80]}...")
    elif status == "skipped":
        reason = feedback.get("reason", "Skipped")
        print(f"      ⚠️  {component}: {reason}")


def _print_tool_checks(tool_checks: Dict[str, Any]) -> None:
    """Print individual tool check results."""
    sql_check = tool_checks.get("sql", {})
    if sql_check.get("correct") is not None:
        icon = "✅" if sql_check["correct"] else "❌"
        print(f"    SQL: {icon}")
        if sql_check.get("query_correct") is not None:
            q_icon = "✅" if sql_check["query_correct"] else "❌"
            print(f"      Query: {q_icon}")
    
    rag_check = tool_checks.get("rag", {})
    if rag_check.get("correct") is not None:
        if rag_check.get("fraction"):
            fraction = rag_check["fraction"]
            score = rag_check.get("score", 0.0)
            icon = "✅" if rag_check["correct"] else "❌"
            print(f"    RAG: {icon} Content: {fraction} ({score:.2f})")
        else:
            icon = "✅" if rag_check["correct"] else "❌"
            print(f"    RAG: {icon}")
    
    api_check = tool_checks.get("api", {})
    if api_check.get("correct") is not None:
        icon = "✅" if api_check["correct"] else "❌"
        print(f"    API: {icon}")
        if not api_check["correct"]:
            _print_api_feedback(api_check)


def _print_api_feedback(api_check: Dict[str, Any]) -> None:
    """Print detailed API tool feedback."""
    mismatch_details = api_check.get("mismatch_details", [])
    match_details = api_check.get("match_details", [])
    matches = api_check.get("matches", 0)
    total_expected = api_check.get("total_expected", 0)
    
    print(f"      Status: {matches}/{total_expected} calls matched")
    
    if mismatch_details:
        print(f"      Mismatches:")
        for mismatch in mismatch_details:
            tool = mismatch.get("tool", "unknown")
            issues = mismatch.get("issues", [])
            expected = mismatch.get("expected_input", {})
            actual = mismatch.get("actual_input", {})
            print(f"        - {tool}:")
            for issue in issues:
                print(f"          {issue}")
            if actual:
                print(f"          Expected input: {expected}")
                print(f"          Actual input: {actual}")
    
    if match_details:
        for detail in match_details:
            print(f"      {detail}")


# ============================================================================
# Main Entry Point
# ============================================================================

async def main():
    """Run evaluation on all composite queries.
    
    Usage:
        python evaluate_agent.py [queries_file] [query_id] [--random N]
    
    Examples:
        python evaluate_agent.py                          # Run all queries
        python evaluate_agent.py composite_queries_extended.jsonl  # Run all queries from file
        python evaluate_agent.py composite_queries_extended.jsonl comp_3300  # Run only query comp_3300
        python evaluate_agent.py comp_3300                 # Run only query comp_3300 (default file)
        python evaluate_agent.py --random 10               # Run 10 random queries from default file
        python evaluate_agent.py composite_queries_extended.jsonl --random 10  # Run 10 random queries from file
    """
    # Parse command line arguments
    queries_file = Path("composite_queries_extended.jsonl")
    query_id = None
    random_count = None
    
    # Check for --random flag
    if "--random" in sys.argv:
        random_idx = sys.argv.index("--random")
        if random_idx + 1 < len(sys.argv):
            try:
                random_count = int(sys.argv[random_idx + 1])
                # Remove --random and its value from argv for easier parsing
                sys.argv = [sys.argv[0]] + [arg for i, arg in enumerate(sys.argv[1:], 1) if i != random_idx and i != random_idx + 1]
            except ValueError:
                print("Error: --random requires a number (e.g., --random 10)")
                sys.exit(1)
        else:
            print("Error: --random requires a number (e.g., --random 10)")
            sys.exit(1)
    
    if len(sys.argv) > 1:
        if len(sys.argv) == 2:
            arg = sys.argv[1]
            queries_file = Path(arg) if Path(arg).exists() else Path("composite_queries_extended.jsonl")
            query_id = arg if not Path(arg).exists() else None
        elif len(sys.argv) >= 3:
            queries_file = Path(sys.argv[1])
            query_id = sys.argv[2]
    
    if not queries_file.exists():
        print(f"Error: {queries_file} not found!")
        sys.exit(1)
    
    # Load queries
    queries = []
    with open(queries_file, "r") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))
    
    # Filter to specific query if requested
    if query_id:
        available_ids = [q.get("id") for q in queries if q.get("id")]
        original_count = len(queries)
        queries = [q for q in queries if q.get("id") == query_id]
        if not queries:
            print(f"Error: Query with id '{query_id}' not found in {queries_file}!")
            print(f"Available query IDs: {available_ids}")
            sys.exit(1)
        print(f"Filtered to query '{query_id}' (1/{original_count} queries)")
    elif random_count:
        # Select random subset
        original_count = len(queries)
        if random_count > original_count:
            print(f"Warning: Requested {random_count} queries but only {original_count} available. Using all {original_count} queries.")
            random_count = original_count
        random.seed()  # Use system time for randomness
        queries = random.sample(queries, random_count)
        print(f"Selected {random_count} random queries from {original_count} total queries")
    
    print(f"Loaded {len(queries)} query/queries from {queries_file}")
    
    # Create session directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = Path("logs") / "session" / f"eval_{timestamp}"
    session_dir.mkdir(parents=True, exist_ok=True)
    print(f"Session directory: {session_dir}")
    
    # Initialize judge LLM
    import os
    judge_model = os.getenv("JUDGE_MODEL", "gpt-4o-mini")
    print(f"\nInitializing judge LLM: {judge_model}")
    judge_llm, _ = init_llm(judge_model)  # Model name not needed for judge
    
    # Build agent graph
    print("Building agent graph...")
    graph = build_agent_graph()
    
    # Run all queries
    print(f"\n{'='*60}")
    print("Running Evaluation")
    print(f"{'='*60}\n")
    
    results = []
    start_time = time.time()
    
    for i, query_data in enumerate(queries, 1):
        result = await run_query(
            query_data=query_data,
            query_num=i,
            total_queries=len(queries),
            graph=graph,
            session_dir=session_dir,
            judge_llm=judge_llm
        )
        results.append(result)
        
        # Print evaluation summary
        if result.get("judge_evaluation"):
            _print_evaluation_summary(result["judge_evaluation"])
    
    elapsed_time = time.time() - start_time
    
    # Calculate and save summary statistics
    summary = _compute_summary_stats(results, timestamp, queries, elapsed_time, session_dir)
    
    # Print final summary
    _print_final_summary(summary, session_dir, elapsed_time, len(queries))


def _compute_summary_stats(
    results: List[Dict[str, Any]],
    timestamp: str,
    queries: List[Dict[str, Any]],
    elapsed_time: float,
    session_dir: Path
) -> Dict[str, Any]:
    """Compute and save summary statistics."""
    valid_results = [r for r in results if r is not None]
    num_valid = len(valid_results)
    
    summary = {
        "session_timestamp": timestamp,
        "total_queries": len(queries),
        "elapsed_time_seconds": elapsed_time,
        "results": results,
        "summary_stats": {
            "average_overall_score": sum(
                (r.get("judge_evaluation") or {}).get("overall_score", 0.0) for r in valid_results
            ) / num_valid if num_valid > 0 else 0.0,
            "average_answer_match_score": sum(
                (r.get("judge_evaluation") or {}).get("answer_match_score", 0.0) for r in valid_results
            ) / num_valid if num_valid > 0 else 0.0,
            "average_tool_call_correctness": sum(
                (r.get("judge_evaluation") or {}).get("tool_call_correctness", 0.0) for r in valid_results
            ) / num_valid if num_valid > 0 else 0.0,
            "average_plan_correctness": sum(
                (r.get("judge_evaluation") or {}).get("plan_correctness", 0.0) for r in valid_results
            ) / num_valid if num_valid > 0 else 0.0,
            "average_replans": sum(
                (r.get("judge_evaluation") or {}).get("replan_count", 0) for r in valid_results
            ) / num_valid if num_valid > 0 else 0.0,
            "queries_with_errors": sum(1 for r in results if r and r.get("error")),
        }
    }
    
    summary_file = session_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    return summary


def _print_final_summary(summary: Dict[str, Any], session_dir: Path, elapsed_time: float, num_queries: int) -> None:
    """Print final summary statistics."""
    stats = summary['summary_stats']
    
    print(f"\n{'='*60}")
    print("Evaluation Summary")
    print(f"{'='*60}")
    print(f"Total queries: {num_queries}")
    print(f"Elapsed time: {elapsed_time:.1f}s")
    print(f"Average Overall Score: {stats['average_overall_score']:.2f}/1.00")
    print(f"Average Answer Match Score: {stats['average_answer_match_score']:.2f}/1.00")
    print(f"Average Plan Correctness: {stats['average_plan_correctness']:.2f}/1.00")
    print(f"Average Tool Call Correctness: {stats['average_tool_call_correctness']:.2f}/1.00")
    print(f"Average Replans: {stats['average_replans']:.2f}")
    print(f"Errors: {stats['queries_with_errors']}")
    print(f"\nResults saved to: {session_dir}")
    print(f"Summary: {session_dir / 'summary.json'}")


if __name__ == "__main__":
    asyncio.run(main())
