# RAG Vector Store Setup

## Overview

The RAG tool now uses **ChromaDB** for persistent on-disk storage of embeddings. This means:
- ✅ Embeddings are created **once** and saved to disk
- ✅ No need to recreate embeddings on every run
- ✅ Fast loading from disk on subsequent runs
- ✅ Vector store persists across sessions

## Setup Steps

### 1. Install Dependencies

Make sure you have the updated requirements:
```bash
pip install -r requirements.txt
```

This includes:
- `chromadb>=0.4.0` - Persistent vector database
- `llama-index-vector-stores-chroma>=0.1.0` - ChromaDB integration for LlamaIndex

### 2. Initialize the Vector Store

**Before using the RAG tool for the first time**, run:

```bash
python init_rag_store.py
```

This script will:
1. Check for your OpenAI API key in `.env`
2. Load documents from the `documents/` folder
3. Create embeddings using OpenAI's embedding model
4. Save the vector store to `chroma_db/` directory
5. Show progress as it indexes

**First run takes a few minutes** (depends on document size and API speed).

### 3. Use the RAG Tool

After initialization, the RAG tool will:
- Automatically load the existing vector store from `chroma_db/`
- No need to recreate embeddings
- Fast query responses

## How It Works

### Storage Location

- **Vector store**: `chroma_db/` directory (created automatically)
- **Documents**: `documents/` folder (your source files)

### Automatic Loading

When `rag_tool()` is called:
1. Checks if `chroma_db/` exists and has embeddings
2. If yes → **loads from disk** (fast, no API calls)
3. If no → **creates new index** (slow, requires API calls)

### Rebuilding the Index

If you update documents in `documents/` folder:

**Option 1**: Delete and recreate
```bash
rm -rf chroma_db/
python init_rag_store.py
```

**Option 2**: Force rebuild (update the script to pass `force_rebuild=True`)

## File Structure

```
multiagent_experiment/
├── documents/              # Source documents
│   ├── music_store_info.txt
│   ├── pricing_policy.txt
│   └── genre_information.txt
├── chroma_db/              # Vector store (created by init_rag_store.py)
│   └── ...                 # ChromaDB internal files
├── init_rag_store.py       # Initialization script
└── agent/
    └── tools_rag.py        # RAG tool with ChromaDB
```

## Troubleshooting

### "OPENAI_API_KEY not found"
- Make sure you have a `.env` file with `OPENAI_API_KEY=your_key`

### "No documents found"
- Check that `documents/` folder exists and contains `.txt` files

### "Vector store not loading"
- Make sure `chroma_db/` directory exists
- Try running `init_rag_store.py` again

### Slow first query
- First query after loading may be slower (warmup)
- Subsequent queries should be fast

## Benefits of Persistent Storage

1. **Cost Savings**: Embeddings created once, reused many times
2. **Speed**: Loading from disk is much faster than creating embeddings
3. **Reliability**: No risk of losing embeddings if process crashes
4. **Scalability**: Can add more documents without full rebuild (with proper implementation)

## Notes

- The `chroma_db/` folder is gitignored (don't commit it)
- Embeddings are specific to the OpenAI model used
- If you change the embedding model, you'll need to rebuild
- Chunk size (512 chars) is fixed - changing it requires rebuild

