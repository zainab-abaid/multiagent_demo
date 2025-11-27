"""Utility functions for LLM initialization supporting both OpenAI and Groq."""

import os
from langchain.chat_models import init_chat_model
from typing import Optional, Any


def init_llm(model_name: str, api_key: Optional[str] = None) -> tuple[Any, str]:
    """
    Initialize an LLM model, supporting both OpenAI and Groq.
    
    Args:
        model_name: Model name. For Groq, use format "groq/<model>" (e.g., "groq/llama-3.1-70b-versatile")
                   For OpenAI, use standard model names (e.g., "gpt-4o-mini")
        api_key: Optional API key. If not provided, will use GROQ_API_KEY or OPENAI_API_KEY from env
    
    Returns:
        Tuple of (Initialized LLM model instance, actual model name string)
    """
    # Check if this is a Groq model
    if model_name.startswith("groq/"):
        # Extract the actual model name (e.g., "llama-3.1-70b-versatile")
        groq_model = model_name.replace("groq/", "")
        
        # Get API key - prefer provided, then env
        groq_api_key = api_key or os.getenv("GROQ_API_KEY")
        
        if not groq_api_key:
            raise ValueError(
                "GROQ_API_KEY environment variable is required for Groq models. "
                "Set it in your .env file or environment."
            )
        
        # Try to import ChatGroq from langchain-groq (preferred method)
        try:
            from langchain_groq import ChatGroq
            llm = ChatGroq(model=groq_model, groq_api_key=groq_api_key)
            return llm, model_name  # Return with groq/ prefix
        except ImportError:
            # Fallback: Use OpenAI-compatible endpoint with ChatOpenAI
            # This works because Groq's API is OpenAI-compatible
            try:
                from langchain_openai import ChatOpenAI
                llm = ChatOpenAI(
                    model=groq_model,
                    openai_api_key=groq_api_key,
                    base_url="https://api.groq.com/openai/v1"
                )
                return llm, model_name  # Return with groq/ prefix
            except ImportError:
                # Last resort: try init_chat_model with custom config
                # Some LangChain versions support this
                llm = init_chat_model(
                    model_name,
                    api_key=groq_api_key,
                    base_url="https://api.groq.com/openai/v1"
                )
                return llm, model_name  # Return with groq/ prefix
    else:
        # Standard OpenAI or other provider model
        # init_chat_model should handle this automatically
        llm = init_chat_model(model_name)
        return llm, model_name

