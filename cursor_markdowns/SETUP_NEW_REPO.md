# Setting Up a New GitHub Repository

## ✅ Verification: debug_agent.py is Compatible

The `debug_agent.py` file has been verified:
- ✅ It doesn't reference `sql_result` (which we removed)
- ✅ It only uses generic state fields (`plan`, `answer_draft`, `latest_result`, `trajectory`)
- ✅ All these fields are still present in the updated `AgentState`
- ✅ It compiles without errors

## Files to Include in Repository

### Essential Files (Must Include):
```
agent/                      # Complete agent directory
  ├── __init__.py
  ├── graph.py
  ├── state.py
  ├── tracing.py
  ├── nodes_*.py            # All node files
  └── tools_*.py            # All tool files
debug_agent.py              # Debug tool
evaluate_agent.py           # Evaluation framework
composite_queries.jsonl     # Ground truth data
init_rag_store.py           # RAG initialization
documents/                  # RAG documents
  ├── music_store_info.txt
  ├── pricing_policy.txt
  └── genre_information.txt
requirements.txt            # Dependencies
.gitignore                  # Git ignore rules
README.md                   # Setup instructions (see README_MINIMAL.md)
```

### Files to Exclude (Already in .gitignore):
- `logs/` - Evaluation logs (generated)
- `chroma_db/` - RAG vector store (regenerated)
- `__pycache__/` - Python cache
- `.env` - API keys (security)
- `Chinook.db` - Database (auto-downloaded if missing)
- All documentation markdown files except README.md

## Step-by-Step Setup

### 1. Create a New Git Repository

```bash
cd /Users/zainababaid/genie/multiagent_experiment

# Initialize git repository
git init

# Or if starting fresh in a new directory:
# mkdir multiagent_minimal && cd multiagent_minimal
# git init
# Copy only the essential files listed above
```

### 2. Add Essential Files

```bash
# Add core agent code
git add agent/

# Add main scripts
git add debug_agent.py
git add evaluate_agent.py

# Add ground truth and supporting files
git add composite_queries.jsonl
git add init_rag_store.py
git add documents/
git add requirements.txt
git add .gitignore

# Add README (use README_MINIMAL.md as template)
cp README_MINIMAL.md README.md
git add README.md
```

### 3. Create Initial Commit

```bash
git commit -m "Initial commit: Multi-agent system with SQL, RAG, and API tools

- Agent implementation with LangGraph
- SQL, RAG, and API tool integrations
- Evaluation framework with ground truth comparison
- Interactive debug tool"
```

### 4. Create GitHub Repository and Push

**Step 4a: Create repository on GitHub (via web browser)**

1. Go to https://github.com/new
2. Choose a repository name (e.g., `multiagent-system`)
3. **Important:** Do NOT check "Initialize this repository with a README" (you already have files)
4. Click "Create repository"

**Step 4b: Connect local repository to GitHub**

```bash
# Add GitHub repository as remote (replace with your actual repo URL)
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# Rename branch to main (if needed)
git branch -M main

# Push to GitHub (you'll be prompted for authentication)
git push -u origin main
```

**Step 4c: Authentication**

When you run `git push`, you'll be prompted for credentials. Options:

1. **GitHub CLI (recommended):**
   ```bash
   gh auth login
   git push -u origin main
   ```

2. **Personal Access Token:**
   - Create token at: https://github.com/settings/tokens
   - Select scope: `repo`
   - When prompted for password, paste the token

3. **SSH (if you prefer):**
   - Use SSH URL: `git@github.com:YOUR_USERNAME/YOUR_REPO_NAME.git`
   - Requires SSH key set up on GitHub

## Quick Verification

After setting up, verify everything works:

```bash
# 1. Create .env file with your API key
echo "OPENAI_API_KEY=your_key" > .env

# 2. Install dependencies
pip install -r requirements.txt

# 3. Initialize RAG store
python init_rag_store.py

# 4. Test debug agent (should work - verified compatible)
python debug_agent.py "How many invoices are there?"

# 5. Test evaluation
python evaluate_agent.py composite_queries.jsonl comp_9
```

## Notes

- The `.gitignore` file will automatically exclude logs, cache files, and generated data
- `Chinook.db` will be auto-downloaded on first SQL query if not present
- `chroma_db/` will be regenerated when you run `init_rag_store.py`
- All agent code uses only the current `AgentState` structure (no deprecated fields)
