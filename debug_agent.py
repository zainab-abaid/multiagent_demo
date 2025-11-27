#!/usr/bin/env python
"""Debug agent with interactive loop and detailed tracing output."""

import sys
import json
import time
import asyncio
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

from agent.graph import build_agent_graph
from agent.state import create_initial_state, AgentConfig

# Load environment variables
load_dotenv()


def format_event(event: dict) -> str:
    """Format a trace event for display."""
    node = event.get("node_name", "unknown")
    event_type = event.get("event_type", "unknown")
    latency = event.get("latency_ms")
    tokens = event.get("total_tokens")
    
    parts = [f"[{node}:{event_type}]"]
    
    if latency is not None:
        parts.append(f"latency: {latency:.1f}ms")
    if tokens is not None:
        parts.append(f"tokens: {tokens}")
    
    if event.get("error"):
        parts.append(f"ERROR: {event['error']}")
    
    return " ".join(parts)


def print_state_summary(state: dict, step_name: str = ""):
    """Print a summary of the current agent state."""
    if step_name:
        print(f"\n{'='*60}")
        print(f"STEP: {step_name}")
        print(f"{'='*60}")
    
    if state.get("plan"):
        plan = state["plan"]
        steps = plan.get("steps", [])
        cursor = state.get("step_cursor", 0)
        print(f"\n[PLAN] {len(steps)} steps, at step {cursor}")
        for i, step in enumerate(steps):
            marker = "->" if i == cursor else "  "
            action = step.get("action_type", "unknown")
            tool = step.get("tool", "")
            desc = step.get("description", "")[:50]
            print(f"  {marker} {i+1}. [{action}] {tool} - {desc}")
    
    if state.get("latest_result"):
        result = state["latest_result"]
        result_str = str(result)[:200]
        print(f"\n[LATEST RESULT] {result_str}...")
    
    if state.get("answer_draft"):
        print(f"\n[ANSWER DRAFT]\n{state['answer_draft']}")


def save_trajectory_to_log(state: dict, log_file: Path):
    """Append trajectory events to log file."""
    with open(log_file, "a") as f:
        for event in state.get("trajectory", []):
            f.write(json.dumps(event) + "\n")


async def main_async():
    """Run the agent in an interactive debug loop (async)."""
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"debug_session_{timestamp}.jsonl"
    
    print(f"\n{'='*60}")
    print("DEBUG AGENT - Interactive Mode")
    print(f"{'='*60}")
    print(f"Log file: {log_file}")
    print(f"Type 'exit' or 'quit' to stop\n")
    
    # Build graph once
    graph = build_agent_graph()
    
    try:
        while True:
            # Get user query
            if len(sys.argv) > 1:
                # If query provided as argument, use it once
                user_query = sys.argv[1]
                print(f"\n[QUERY] {user_query}\n")
            else:
                # Interactive mode
                user_query = input("\nEnter your query (or 'exit' to quit): ").strip()
                if not user_query or user_query.lower() in ['exit', 'quit']:
                    print("\nExiting...")
                    break
            
            print(f"\n{'='*60}")
            print(f"PROCESSING: {user_query}")
            print(f"{'='*60}\n")
            
            # Create initial state
            config = AgentConfig(model_name="gpt-4o-mini")
            initial_state = create_initial_state(user_query, config)
            
            # Track previous trajectory length to show new events
            prev_trajectory_len = 0
            
            try:
                # Run the graph and stream intermediate states (async)
                final_state = None
                step_count = 0
                
                async for step in graph.astream(initial_state, stream_mode="values"):
                    final_state = step
                    step_count += 1
                    
                    # Show new events since last step
                    trajectory = step.get("trajectory", [])
                    new_events = trajectory[prev_trajectory_len:]
                    
                    if new_events:
                        print(f"\n--- Step {step_count} Events ---")
                        for event in new_events:
                            print(f"  {format_event(event)}")
                    
                    prev_trajectory_len = len(trajectory)
                    
                    # Show state summary at key points
                    if step.get("plan") and step.get("step_cursor", 0) == 0:
                        print_state_summary(step, "After Planning")
                    elif step.get("latest_result"):
                        print_state_summary(step, "After Tool Call")
                    elif step.get("answer_draft"):
                        print_state_summary(step, "After Answer Generation")
                
                if final_state is None:
                    final_state = initial_state
                
                # Print final results
                print(f"\n{'='*60}")
                print("FINAL RESULTS")
                print(f"{'='*60}")
                
                answer = final_state.get("answer_draft", "No answer generated.")
                print(f"\n[FINAL ANSWER]\n{answer}\n")
                
                # Print trajectory summary
                trajectory = final_state.get("trajectory", [])
                print(f"\n[TRACE SUMMARY]")
                print(f"  Total events: {len(trajectory)}")
                
                llm_calls = [e for e in trajectory if e.get("event_type") == "llm_call"]
                tool_calls = [e for e in trajectory if e.get("event_type") == "tool_call"]
                
                print(f"  LLM calls: {len(llm_calls)}")
                print(f"  Tool calls: {len(tool_calls)}")
                
                total_tokens = sum(e.get("total_tokens", 0) or 0 for e in llm_calls)
                total_latency = sum(e.get("latency_ms", 0) or 0 for e in trajectory)
                
                if total_tokens > 0:
                    print(f"  Total tokens: {total_tokens}")
                if total_latency > 0:
                    print(f"  Total latency: {total_latency:.1f}ms ({total_latency/1000:.2f}s)")
                
                # Save trajectory to log
                save_trajectory_to_log(final_state, log_file)
                print(f"\n[LOG] Trajectory saved to {log_file}")
                
                # If query was from command line, exit after one run
                if len(sys.argv) > 1:
                    break
                    
            except Exception as e:
                print(f"\n[ERROR] {e}")
                import traceback
                traceback.print_exc()
                
                # Still save what we have
                if final_state:
                    save_trajectory_to_log(final_state, log_file)
                
                # If query was from command line, exit on error
                if len(sys.argv) > 1:
                    break
                    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        import traceback
        traceback.print_exc()


def main():
    """Synchronous entry point that runs the async main."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()

