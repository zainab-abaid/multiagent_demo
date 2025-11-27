"""RAG tool with LlamaIndex and ChromaDB persistent vector store."""

import os
from pathlib import Path
from typing import Optional
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# Global index (initialized on first use)
_vector_index = None
_documents_path = None
_chroma_path = "chroma_db"  # Directory where ChromaDB stores data


def _init_rag_index(documents_path: str = "documents", chunk_size: int = 512, force_rebuild: bool = False):
    """
    Initialize the RAG vector store index with persistent ChromaDB storage.
    
    If the vector store already exists on disk, it will be loaded.
    Otherwise, documents will be indexed and embeddings created.
    
    Parameters
    ----------
    documents_path : str
        Path to directory containing documents to index
    chunk_size : int
        Fixed chunk size for text splitting
    force_rebuild : bool
        If True, rebuild the index even if it exists on disk
        
    Returns
    -------
    VectorStoreIndex
        The initialized vector store index
    """
    global _vector_index, _documents_path, _chroma_path
    
    # Check if already initialized with same path
    if _vector_index is not None and _documents_path == documents_path and not force_rebuild:
        return _vector_index
    
    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found in environment. "
            "Please set it in .env file or export it."
        )
    
    # Set up OpenAI embedding model
    Settings.embed_model = OpenAIEmbedding(api_key=api_key)
    
    # Set up node parser with fixed chunk size
    node_parser = SimpleNodeParser.from_defaults(chunk_size=chunk_size, chunk_overlap=20)
    Settings.node_parser = node_parser
    
    # Initialize ChromaDB client
    chroma_path = Path(_chroma_path)
    chroma_path.mkdir(exist_ok=True)
    
    # Create or get ChromaDB collection
    chroma_client = chromadb.PersistentClient(path=str(chroma_path))
    chroma_collection = chroma_client.get_or_create_collection(
        name="documents",
        metadata={"hnsw:space": "cosine"}
    )
    
    # Create vector store from ChromaDB collection
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Check if collection is empty (needs indexing)
    collection_count = chroma_collection.count()
    
    if collection_count == 0 or force_rebuild:
        # Need to index documents
        print(f"[RAG] Indexing documents from {documents_path}...")
        
        # Load documents
        doc_path = Path(documents_path)
        if not doc_path.exists():
            raise ValueError(f"Documents path does not exist: {documents_path}")
        
        reader = SimpleDirectoryReader(input_dir=str(doc_path))
        documents = reader.load_data()
        
        if not documents:
            raise ValueError(f"No documents found in {documents_path}")
        
        # Create index from documents (this will create embeddings and store in ChromaDB)
        _vector_index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True
        )
        
        print(f"[RAG] Indexed {len(documents)} documents. Embeddings saved to {chroma_path}")
    else:
        # Load existing index from ChromaDB
        print(f"[RAG] Loading existing vector store from {chroma_path} ({collection_count} embeddings)...")
        _vector_index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context
        )
        print(f"[RAG] Vector store loaded successfully")
    
    _documents_path = documents_path
    
    return _vector_index


def rag_tool(query: str, top_k: int = 5, documents_path: str = "documents") -> list[dict]:
    """
    RAG retrieval using LlamaIndex vector store.
    
    Parameters
    ----------
    query : str
        Search query
    top_k : int
        Number of documents to retrieve
    documents_path : str
        Path to documents directory (default: "documents")
        
    Returns
    -------
    list[dict]
        List of document dicts with keys: "content", "metadata", "score"
    """
    try:
        # Initialize index if needed
        index = _init_rag_index(documents_path)
        
        # Create query engine
        query_engine = index.as_query_engine(similarity_top_k=top_k)
        
        # Perform query
        response = query_engine.query(query)
        
        # Extract results
        results = []
        if hasattr(response, 'source_nodes') and response.source_nodes:
            for node in response.source_nodes:
                result = {
                    "content": node.text,
                    "score": node.score if hasattr(node, 'score') else None,
                    "metadata": {
                        "node_id": node.node_id if hasattr(node, 'node_id') else None,
                        "file_path": getattr(node, 'file_path', None),
                    }
                }
                results.append(result)
        else:
            # Fallback: return the response text
            results.append({
                "content": str(response),
                "score": None,
                "metadata": {}
            })
        
        return results
        
    except Exception as e:
        # Return error in result format
        return [{
            "content": f"Error during RAG retrieval: {str(e)}",
            "score": None,
            "metadata": {"error": True}
        }]

