"""RAG tool with LlamaIndex and ChromaDB persistent vector store."""

import os
from pathlib import Path
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

_vector_index = None
_documents_path = None
_chroma_path = "chroma_db"


def _init_rag_index(documents_path: str = "documents", chunk_size: int = None, force_rebuild: bool = False):
    """Initialize RAG vector store index with persistent ChromaDB storage."""
    global _vector_index, _documents_path, _chroma_path
    
    if chunk_size is None:
        chunk_size = int(os.getenv("RAG_CHUNK_SIZE", "512"))
    
    if _vector_index is not None and _documents_path == documents_path and not force_rebuild:
        return _vector_index
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment. Please set it in .env file or export it.")
    
    Settings.embed_model = OpenAIEmbedding(api_key=api_key)
    node_parser = SimpleNodeParser.from_defaults(chunk_size=chunk_size, chunk_overlap=20)
    Settings.node_parser = node_parser
    
    chroma_path = Path(_chroma_path)
    chroma_path.mkdir(exist_ok=True)
    
    chroma_client = chromadb.PersistentClient(path=str(chroma_path))
    chroma_collection = chroma_client.get_or_create_collection(
        name="documents",
        metadata={"hnsw:space": "cosine"}
    )
    
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    collection_count = chroma_collection.count()
    
    if collection_count == 0 or force_rebuild:
        print(f"[RAG] Indexing documents from {documents_path}...")
        
        doc_path = Path(documents_path)
        if not doc_path.exists():
            raise ValueError(f"Documents path does not exist: {documents_path}")
        
        reader = SimpleDirectoryReader(input_dir=str(doc_path))
        documents = reader.load_data()
        
        if not documents:
            raise ValueError(f"No documents found in {documents_path}")
        
        _vector_index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True
        )
        
        print(f"[RAG] Indexed {len(documents)} documents. Embeddings saved to {chroma_path}")
    else:
        print(f"[RAG] Loading existing vector store from {chroma_path} ({collection_count} embeddings)...")
        _vector_index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context
        )
        print(f"[RAG] Vector store loaded successfully")
    
    _documents_path = documents_path
    
    return _vector_index


def rag_tool(query: str, top_k: int = None, documents_path: str = None) -> list[dict]:
    """
    RAG retrieval using LlamaIndex vector store.
    
    Once ChromaDB is initialized (via init_rag_store.py), it loads from the existing vector store.
    documents_path is only needed if ChromaDB is empty (for initial indexing).
    """
    if top_k is None:
        top_k = int(os.getenv("RAG_TOP_K", "5"))
    
    # documents_path only needed if ChromaDB is empty (for initial indexing)
    # Once vector store exists, it loads from ChromaDB regardless
    if documents_path is None:
        documents_path = "documents"
    
    try:
        index = _init_rag_index(documents_path)
        query_engine = index.as_query_engine(similarity_top_k=top_k)
        response = query_engine.query(query)
        
        results = []
        if hasattr(response, 'source_nodes') and response.source_nodes:
            for node in response.source_nodes:
                results.append({
                    "content": node.text,
                    "score": node.score if hasattr(node, 'score') else None,
                    "metadata": {
                        "node_id": node.node_id if hasattr(node, 'node_id') else None,
                        "file_path": getattr(node, 'file_path', None),
                    }
                })
        else:
            results.append({
                "content": str(response),
                "score": None,
                "metadata": {}
            })
        
        return results
        
    except Exception as e:
        return [{
            "content": f"Error during RAG retrieval: {str(e)}",
            "score": None,
            "metadata": {"error": True}
        }]

