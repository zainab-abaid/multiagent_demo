"""LangGraph workflow construction."""

from langgraph.graph import StateGraph, START, END
from typing import Any

from agent.state import AgentState
from agent.nodes_input import input_node
from agent.nodes_planner import planner_node
from agent.nodes_controller import plan_controller_node, route_controller
from agent.nodes_tools import tool_caller_node
from agent.nodes_answer import answer_node
from agent.nodes_reflection import reflection_node


def build_agent_graph() -> Any:
    """
    Build and return the compiled LangGraph agent.
    
    Flow:
    1. input_node: Initialize state
    2. planner_node: Create execution plan
    3. Loop:
       - plan_controller_node: Route to next step
       - tool_caller_node: Execute tools (loops back to controller)
    4. answer_node: Generate final answer
    5. reflection_node: Reflect on episode
    6. END
    """
    # Create graph with AgentState
    builder = StateGraph(AgentState)
    
    # Add nodes
    builder.add_node("input", input_node)
    builder.add_node("planner", planner_node)
    builder.add_node("plan_controller", plan_controller_node)
    builder.add_node("tool_caller", tool_caller_node)
    builder.add_node("answer", answer_node)
    builder.add_node("reflection", reflection_node)
    
    # Wire the graph
    # Start -> input
    builder.add_edge(START, "input")
    
    # input -> planner
    builder.add_edge("input", "planner")
    
    # planner -> plan_controller
    builder.add_edge("planner", "plan_controller")
    
    # plan_controller routes conditionally
    builder.add_conditional_edges(
        "plan_controller",
        route_controller,  # routing function (separate from the node)
        {
            "tool_caller": "tool_caller",
            "planner": "planner",  # Loop back to planner for re-evaluation
            "answer": "answer",
            "end": END,
        }
    )
    
    # tool_caller loops back to plan_controller
    builder.add_edge("tool_caller", "plan_controller")
    
    # answer -> reflection
    builder.add_edge("answer", "reflection")
    
    # reflection -> end
    builder.add_edge("reflection", END)
    
    # Compile and return
    return builder.compile()

