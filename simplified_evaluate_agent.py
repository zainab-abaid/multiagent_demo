#!/usr/bin/env python
"""Simplified evaluation script - only checks answer matching."""

import json
import sys
import time
import asyncio
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
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
- **CRITICAL for null handling**: 
  * If the expected value is `null` (JSON null), look for phrases like "not available", "None", "null", "no data", "result is None", "does not exist", "could not find", etc.
  * When you see these phrases for a key that expects null, extract `null` (JSON null, not the string "null")
  * For example, if expected has `"result": null` and agent says "result is None" or "not available", extract `"result": null`
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


# ============================================================================
# Utilities
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


async def _invoke_llm(llm, messages):
    """Invoke LLM and return response."""
    if hasattr(llm, 'ainvoke'):
        return await llm.ainvoke(messages)
    else:
        return llm.invoke(messages)


async def _parse_llm_json_response(response):
    """Parse JSON from LLM response, handling markdown code blocks."""
    text = response.content if hasattr(response, 'content') else str(response)
    text = text.strip()
    
    # Remove markdown code blocks if present
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
        text = text.replace("```json", "").replace("```", "").strip()
    
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


async def extract_answer_components(
    question: str,
    agent_answer: str,
    expected_answer: Dict[str, Any],
    judge_llm
) -> Dict[str, Any]:
    """Extract answer components from agent's answer using LLM."""
    if not expected_answer:
        return {}
    
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
- **CRITICAL**: If the expected value is `null` and the agent indicates the data is not available (e.g., "not available", "None", "no data", "result is None"), extract `null` (JSON null, not the string "null")

Return ONLY a JSON object with exactly these keys."""
    
    try:
        messages = [
            SystemMessage(content=PROMPT_EXTRACT_COMPONENTS),
            HumanMessage(content=user_prompt)
        ]
        response = await _invoke_llm(judge_llm, messages)
        extracted = await _parse_llm_json_response(response)
        
        # Ensure all expected keys are present
        return {key: extracted.get(key, None) for key in expected_answer.keys()}
    except Exception as e:
        print(f"  [WARNING] Failed to extract components: {e}")
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


def _numeric_match(expected: float, actual: Any, base_tolerance: float = 2.0) -> bool:
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
        return {"score": 1.0, "reasoning": "No expected answer provided - assuming correct"}
    
    correct = 0
    total = len(expected)
    reasoning_parts = []
    
    for key in expected:
        exp_val = expected[key]
        got_val = extracted.get(key, None)
        
        # Handle null/None values specially
        if exp_val is None:
            # Expected value is null - check if agent indicated unavailability
            if got_val is None or (isinstance(got_val, str) and got_val.lower() in ["null", "none", "n/a", "not available", "no data"]):
                correct += 1
                reasoning_parts.append(f"  - {key}: CORRECT (expected: null, agent indicated unavailability)")
            else:
                reasoning_parts.append(f"  - {key}: INCORRECT (expected: null, got: {str(got_val)[:100]})")
            continue
        
        if got_val is None:
            reasoning_parts.append(f"  - {key}: MISSING (expected: {str(exp_val)[:100]})")
            continue
        
        # Numeric comparison
        if isinstance(exp_val, (int, float)):
            if _numeric_match(exp_val, got_val, numeric_tolerance):
                correct += 1
                reasoning_parts.append(f"  - {key}: CORRECT (expected: {exp_val}, got: {got_val})")
            else:
                got_num = float(got_val)
                exp_num = float(exp_val)
                diff = abs(got_num - exp_num)
                tolerance = max(numeric_tolerance, abs(exp_num) * 0.02) if exp_num > 10 else numeric_tolerance
                reasoning_parts.append(f"  - {key}: INCORRECT (expected: {exp_val}, got: {got_val}, diff: {diff:.2f}, tolerance: {tolerance:.2f})")
        else:
            # String comparison - use LLM for semantic matching
            is_correct = await judge_string_match(question, str(exp_val), str(got_val), key, judge_llm)
            if is_correct:
                correct += 1
                reasoning_parts.append(f"  - {key}: CORRECT (semantically equivalent)")
            else:
                reasoning_parts.append(f"  - {key}: INCORRECT (expected: {str(exp_val)[:100]}, got: {str(got_val)[:100]})")
    
    score = correct / total if total > 0 else 0.0
    reasoning = "\n".join(reasoning_parts) if reasoning_parts else "All components correct"
    
    return {
        "score": score,
        "correct_count": correct,
        "total_count": total,
        "reasoning": reasoning
    }


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
    """Run a single query through the agent and evaluate answer matching."""
    query_id = query_data["id"]
    question = query_data["question"]
    expected_answer = query_data.get("expected_answer", {})
    
    print(f"\n{'='*60}")
    print(f"[{query_num}/{total_queries}] Query: {query_id}")
    print(f"{'='*60}")
    print(f"\nInput Query: {question}")
    print(f"\nGround Truth Answer:")
    print(json.dumps(expected_answer, indent=2))
    
    # Create log directory for this query
    query_log_dir = session_dir / f"query{query_num}"
    query_log_dir.mkdir(parents=True, exist_ok=True)
    
    # Save query metadata
    with open(query_log_dir / "query_metadata.json", "w") as f:
        json.dump({
            "query_id": query_id,
            "question": question,
            "expected_answer": expected_answer,
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
        print(f"\n[ERROR] {e}")
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
            "answer_match_score": 0.0,
            "reasoning": f"Error: {str(e)}"
        }
    
    if final_state is None:
        final_state = initial_state
    
    # Save final state (trajectory already saved incrementally)
    with open(query_log_dir / "final_state.json", "w") as f:
        json.dump(final_state, f, indent=2, default=str)
    
    # Get agent's answer
    agent_answer = final_state.get("answer_draft", "No answer generated.")
    print(f"\nAgent Answer:\n{agent_answer}")
    
    # Evaluate answer matching
    extracted = await extract_answer_components(question, agent_answer, expected_answer, judge_llm)
    answer_score = await score_answer_correctness(
        expected_answer,
        extracted,
        question,
        agent_answer,
        judge_llm
    )
    
    answer_match_score = answer_score["score"]
    reasoning = answer_score.get("reasoning", "")
    
    print(f"\nAnswer Match Score: {answer_match_score:.2f}/1.00")
    if reasoning:
        print(f"\nReasoning:\n{reasoning}")
    
    # Save evaluation result
    judge_evaluation = {
        "answer_match_score": answer_match_score,
        "component_feedback": answer_score.get("reasoning", ""),
        "extracted": extracted
    }
    
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
        "expected_answer": expected_answer,
        "agent_answer": agent_answer,
        "answer_match_score": answer_match_score,
        "reasoning": reasoning
    }


# ============================================================================
# Main
# ============================================================================

async def main():
    """Main evaluation loop."""
    print("Simplified Evaluation - Answer Matching Only")
    print("=" * 60)
    
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
            print(f"Available query IDs: {available_ids[:10]}...")  # Show first 10
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
    
    # Initialize judge LLM
    import os
    judge_model = os.getenv("JUDGE_MODEL", "gpt-4o-mini")
    print(f"\nInitializing judge LLM: {judge_model}")
    judge_llm, _ = init_llm(judge_model)
    
    # Build agent graph
    print("Building agent graph...")
    graph = build_agent_graph()
    
    # Create session directory for logs
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = Path("logs") / "session" / f"eval_{session_timestamp}"
    session_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nLogs will be saved to: {session_dir}")
    
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
    
    elapsed_time = time.time() - start_time
    
    # Save summary
    summary = {
        "total_queries": len(results),
        "elapsed_time_seconds": elapsed_time,
        "average_answer_match_score": None,
        "results": results
    }
    
    if results:
        answer_scores = [r["answer_match_score"] for r in results if r.get("answer_match_score") is not None]
        if answer_scores:
            avg_score = sum(answer_scores) / len(answer_scores)
            summary["average_answer_match_score"] = avg_score
    
    with open(session_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    
    if results:
        answer_scores = [r["answer_match_score"] for r in results if r.get("answer_match_score") is not None]
        if answer_scores:
            avg_score = sum(answer_scores) / len(answer_scores)
            print(f"\nAverage Answer Match Score: {avg_score:.2f}/1.00")
            print(f"Total Queries: {len(results)}")
            print(f"Total Time: {elapsed_time:.2f} seconds")
            print(f"\nLogs saved to: {session_dir}")
        else:
            print("\nNo valid answer match scores to average.")
    else:
        print("\nNo results to summarize.")


if __name__ == "__main__":
    asyncio.run(main())

