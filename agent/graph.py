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
    1. START -> input: Initialize state
    2. input -> planner: Create initial execution plan
    3. planner -> plan_controller: Route to next step
    4. plan_controller conditionally routes to:
       - tool_caller: Execute tools (loops back to plan_controller)
       - planner: Replan if needed (plan_controller redirects back to planner for maximum MAX_TURNS (in .env) times)
       - answer: Generate final answer (triggered when planner indicates ready to answer)
       - end: End immediately if done
    5. answer -> reflection: Reflect on episode
    6. reflection -> END
    """
    builder = StateGraph(AgentState)
    
    builder.add_node("input", input_node)
    builder.add_node("planner", planner_node)
    builder.add_node("plan_controller", plan_controller_node)
    builder.add_node("tool_caller", tool_caller_node)
    builder.add_node("answer", answer_node)
    builder.add_node("reflection", reflection_node)
    
    builder.add_edge(START, "input")
    builder.add_edge("input", "planner")
    builder.add_edge("planner", "plan_controller")
    
    builder.add_conditional_edges(
        "plan_controller",
        route_controller,
        {
            "tool_caller": "tool_caller",
            "planner": "planner",
            "answer": "answer",
            "end": END,
        }
    )
    
    builder.add_edge("tool_caller", "plan_controller")
    builder.add_edge("answer", "reflection")
    builder.add_edge("reflection", END)
    
    return builder.compile()

