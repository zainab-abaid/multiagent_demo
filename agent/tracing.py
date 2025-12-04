"""Tracing utilities for LLM and tool calls."""

import time
import uuid
from typing import Any, Callable
from langchain_core.messages import SystemMessage, HumanMessage
from agent.state import AgentState, TraceEvent, event_to_dict


def create_trace_event(
    node_name: str,
    event_type: str,
    state: AgentState,
    input_data: dict | str | None = None,
    output_data: dict | str | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_tokens: int | None = None,
    latency_ms: float | None = None,
    tool_name: str | None = None,
    model_name: str | None = None,
    error: str | None = None,
) -> dict:
    """Create a TraceEvent and append it to state.trajectory."""
    event = TraceEvent(
        event_id=str(uuid.uuid4()),
        timestamp=time.time(),
        node_name=node_name,
        event_type=event_type,
        input=input_data,
        output=output_data,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        latency_ms=latency_ms,
        tool_name=tool_name,
        model_name=model_name,
        error=error,
    )
    event_dict = event_to_dict(event)
    if "trajectory" not in state:
        state["trajectory"] = []
    state["trajectory"].append(event_dict)
    return event_dict


async def traced_llm_call(
    node_name: str,
    state: AgentState,
    llm_callable: Callable[..., Any],
    llm_input: dict | str,
    model_name: str | None = None,
    is_sync: bool = False,
    **kwargs
) -> Any:
    """
    Call the given llm_callable, measure latency and token usage (if available),
    append a TraceEvent to state.trajectory, and return the output.
    
    If is_sync is True, llm_callable.invoke is called synchronously.
    Otherwise (default), we assume it can be awaited if needed or use invoke if it's a sync wrapper.
    However, LangChain LLMs are often invoked with .invoke() (sync) or .ainvoke() (async).
    This function wrapper is async, so it's best suited for async callers.
    But if the underlying call is sync, we can just call .invoke().
    """
    start_time = time.time()
    error = None
    output = None
    prompt_tokens = None
    completion_tokens = None
    total_tokens = None
    
    try:
        # Call the LLM
        # llm_input should be a list of messages or a single message
        if is_sync:
             if isinstance(llm_input, list):
                response = llm_callable.invoke(llm_input)
             elif isinstance(llm_input, dict):
                response = llm_callable.invoke(**llm_input, **kwargs)
             else:
                response = llm_callable.invoke(llm_input, **kwargs)
        else:
            # Prefer ainvoke if available, otherwise invoke
            if hasattr(llm_callable, 'ainvoke'):
                if isinstance(llm_input, list):
                    response = await llm_callable.ainvoke(llm_input)
                elif isinstance(llm_input, dict):
                    response = await llm_callable.ainvoke(**llm_input, **kwargs)
                else:
                    response = await llm_callable.ainvoke(llm_input, **kwargs)
            else:
                 # Fallback to sync invoke if ainvoke missing
                 if isinstance(llm_input, list):
                    response = llm_callable.invoke(llm_input)
                 elif isinstance(llm_input, dict):
                    response = llm_callable.invoke(**llm_input, **kwargs)
                 else:
                    response = llm_callable.invoke(llm_input, **kwargs)
        
        output = response
        
        # Extract token counts from response
        # LangChain responses have usage_metadata as a dict-like object
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            # usage_metadata can be a dict or an object with attributes
            if isinstance(response.usage_metadata, dict):
                prompt_tokens = response.usage_metadata.get('input_tokens')
                completion_tokens = response.usage_metadata.get('output_tokens')
                total_tokens = response.usage_metadata.get('total_tokens')
            else:
                # Try as attributes
                prompt_tokens = getattr(response.usage_metadata, 'input_tokens', None) or getattr(response.usage_metadata, 'prompt_tokens', None)
                completion_tokens = getattr(response.usage_metadata, 'output_tokens', None) or getattr(response.usage_metadata, 'completion_tokens', None)
                total_tokens = getattr(response.usage_metadata, 'total_tokens', None)
        
        # Also check response_metadata as fallback
        if (prompt_tokens is None or completion_tokens is None or total_tokens is None) and hasattr(response, 'response_metadata'):
            metadata = response.response_metadata
            if metadata:
                token_usage = metadata.get('token_usage', {})
                if token_usage:
                    prompt_tokens = prompt_tokens or token_usage.get('prompt_tokens') or token_usage.get('input_tokens')
                    completion_tokens = completion_tokens or token_usage.get('completion_tokens') or token_usage.get('output_tokens')
                    total_tokens = total_tokens or token_usage.get('total_tokens')
            
    except Exception as e:
        error = str(e)
        raise
    finally:
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        
        # Prepare input/output for logging
        # Replace system prompts with placeholders to reduce log size, but keep variables from HumanMessage
        if isinstance(llm_input, list):
            cleaned_messages = []
            for msg in llm_input:
                if isinstance(msg, SystemMessage):
                    content = msg.content
                    # Replace known system prompts with placeholders
                    if content.startswith("You are a planning assistant"):
                        cleaned_messages.append(SystemMessage(content="<planner prompt>"))
                    elif content.startswith("You are a sophisticated planning assistant"):
                        cleaned_messages.append(SystemMessage(content="<replan prompt>"))
                    elif content.startswith("You are an assistant that answers questions by writing SQL"):
                        cleaned_messages.append(SystemMessage(content="<sql_tool prompt>"))
                    elif content.startswith("You are a helpful assistant that answers user questions"):
                        cleaned_messages.append(SystemMessage(content="<answer prompt>"))
                    elif content.startswith("You are a reflection assistant"):
                        cleaned_messages.append(SystemMessage(content="<reflection prompt>"))
                    elif content.startswith("You are an API routing assistant"):
                        cleaned_messages.append(SystemMessage(content="<api_router prompt>"))
                    else:
                        # Unknown prompt - show first 50 chars
                        cleaned_messages.append(SystemMessage(content=f"<system prompt: {content[:50]}...>"))
                elif isinstance(msg, HumanMessage):
                    # Keep HumanMessage as-is (contains the variables)
                    cleaned_messages.append(msg)
                else:
                    cleaned_messages.append(msg)
            input_str = str(cleaned_messages)
        else:
            input_str = str(llm_input) if llm_input else None
        
        # For output, show more (up to 2000 chars) since it's the response
        output_str = str(output)[:2000] if output else None
        
        create_trace_event(
            node_name=node_name,
            event_type="llm_call",
            state=state,
            input_data=input_str,
            output_data=output_str,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            model_name=model_name,
            error=error,
        )
    
    return output


def traced_tool_call(
    node_name: str,
    state: AgentState,
    tool_name: str,
    tool_callable: Callable[..., Any],
    tool_input: dict | str | None = None,
    **kwargs
) -> Any:
    """
    Call the given tool, measure latency, append a TraceEvent, and return the output.
    """
    start_time = time.time()
    error = None
    output = None
    
    try:
        # Call the tool
        if isinstance(tool_input, dict):
            output = tool_callable(**tool_input, **kwargs)
        elif tool_input is not None:
            output = tool_callable(tool_input, **kwargs)
        else:
            output = tool_callable(**kwargs)
            
    except Exception as e:
        error = str(e)
        raise
    finally:
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        
        # Prepare input/output for logging
        # For tool calls, especially SQL, we need more space to see full results
        # Remove 'state' from tool_input if present to avoid circular reference in logs
        if isinstance(tool_input, dict):
            clean_input = {k: v for k, v in tool_input.items() if k != "state"}
            input_str = str(clean_input)[:2000] if clean_input else None
        else:
            input_str = str(tool_input)[:2000] if tool_input else None
        
        # For output, preserve full structure for SQL results
        if tool_name == "sql_tool" and isinstance(output, dict):
            # For SQL, show the full structure but format it nicely
            import json
            try:
                # Create a summary that includes key fields
                sql_summary = {
                    "generated_sql": output.get("generated_sql", "")[:500],  # SQL query can be truncated
                    "query_result": output.get("query_result", output.get("execution_result", {}).get("result", ""))[:1000],  # But result should be visible
                    "success": output.get("success", True),
                }
                output_str = json.dumps(sql_summary, indent=2)[:3000]  # Up to 3000 chars
            except:
                output_str = str(output)[:3000]
        else:
            output_str = str(output)[:3000] if output else None
        
        create_trace_event(
            node_name=node_name,
            event_type="tool_call",
            state=state,
            input_data=input_str,
            output_data=output_str,
            latency_ms=latency_ms,
            tool_name=tool_name,
            error=error,
        )
    
    return output


async def traced_async_tool_call(
    node_name: str,
    state: AgentState,
    tool_name: str,
    tool_callable: Callable[..., Any],
    tool_input: dict | str | None = None,
    **kwargs
) -> Any:
    """
    Async version of traced_tool_call for async tool functions.
    Call the given async tool, measure latency, append a TraceEvent, and return the output.
    """
    start_time = time.time()
    error = None
    output = None
    
    try:
        # Call the async tool
        if isinstance(tool_input, dict):
            output = await tool_callable(**tool_input, **kwargs)
        elif tool_input is not None:
            output = await tool_callable(tool_input, **kwargs)
        else:
            output = await tool_callable(**kwargs)
            
    except Exception as e:
        error = str(e)
        raise
    finally:
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        
        # Prepare input/output for logging
        if isinstance(tool_input, dict):
            clean_input = {k: v for k, v in tool_input.items() if k != "state"}
            input_str = str(clean_input)[:2000] if clean_input else None
        else:
            input_str = str(tool_input)[:2000] if tool_input else None
        
        # For output, format SQL results compactly for planner visibility
        if tool_name == "sql_tool" and isinstance(output, dict):
            try:
                generated_sql = output.get("generated_sql", "")
                query_result = output.get("query_result", output.get("execution_result", {}).get("result", ""))
                success = output.get("success", True)
                # Compact format: prioritize showing query_result (the actual data)
                # SQL query truncated to 100 chars since planner mainly needs to see the result
                sql_short = generated_sql[:100] + "..." if len(generated_sql) > 100 else generated_sql
                output_str = f"SQL: {sql_short} | Result: {query_result} | Success: {success}"
            except:
                output_str = str(output)[:500]
        else:
            output_str = str(output)[:3000] if output else None
        
        create_trace_event(
            node_name=node_name,
            event_type="tool_call",
            state=state,
            input_data=input_str,
            output_data=output_str,
            latency_ms=latency_ms,
            tool_name=tool_name,
            error=error,
        )
    
    return output

