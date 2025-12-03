# Setup Instructions

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

## 2. Set Up Environment Variables

Create a `.env` file in the root directory:

```bash
cp .env.example .env
```

Then edit `.env` and add your OpenAI API key:

```
OPENAI_API_KEY=sk-your-actual-api-key-here
```

## 3. Initialize RAG Vector Store

Before using the RAG tool, you need to create the vector store with embeddings:

```bash
python init_rag_store.py
```

This will:
- Load documents from the `documents/` folder
- Create embeddings using OpenAI (requires API key)
- Save the vector store to `chroma_db/` directory
- Take a few minutes on first run

**Note**: The vector store is persistent - you only need to run this once (or when you add/update documents).

## 4. Verify Documents Folder

The `documents/` folder should contain:
- `music_store_info.txt`
- `pricing_policy.txt`
- `genre_information.txt`

## 5. Run the Debug Agent

```bash
python debug_agent.py
```

Or with a specific query:

```bash
python debug_agent.py "What was the total revenue in USD from Latin tracks sold in 2013, and what would that amount be if converted to EUR based on the store's currency policy?"
```

## Notes

- The RAG vector store is created once with `init_rag_store.py` and reused
- If you update documents, run `init_rag_store.py` again with `force_rebuild=True` or delete `chroma_db/`
- All traces are saved to `logs/debug_session_<timestamp>.jsonl`
- The agent will show intermediate steps and final results in the terminal

