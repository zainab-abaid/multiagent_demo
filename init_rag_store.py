#!/usr/bin/env python
"""Initialize the RAG vector store by indexing documents."""

import os
from pathlib import Path
from dotenv import load_dotenv
from agent.tools_rag import _init_rag_index

# Load environment variables
load_dotenv()


def main():
    """Initialize the RAG vector store."""
    print("=" * 60)
    print("RAG Vector Store Initialization")
    print("=" * 60)
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n[ERROR] OPENAI_API_KEY not found in environment.")
        print("Please create a .env file with your OpenAI API key:")
        print("  OPENAI_API_KEY=your_key_here\n")
        return
    
    documents_path = "documents"
    
    # Check if documents directory exists
    if not Path(documents_path).exists():
        print(f"\n[ERROR] Documents directory '{documents_path}' does not exist.")
        print("Please create it and add your text files.\n")
        return
    
    print(f"\n[INFO] Documents path: {documents_path}")
    print(f"[INFO] Vector store will be saved to: chroma_db/")
    print(f"[INFO] This may take a few minutes to create embeddings...\n")
    
    try:
        # Initialize the index (will create embeddings if needed)
        index = _init_rag_index(documents_path=documents_path, force_rebuild=True)
        
        print("\n" + "=" * 60)
        print("[SUCCESS] RAG vector store initialized successfully!")
        print("=" * 60)
        print("\nYou can now use the RAG tool. The embeddings are saved on disk")
        print("and will be reused in future runs.\n")
        
    except Exception as e:
        print(f"\n[ERROR] Failed to initialize RAG vector store: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

