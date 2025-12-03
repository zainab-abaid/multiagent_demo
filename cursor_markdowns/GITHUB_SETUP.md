# GitHub Repository Setup Guide

This guide walks you through setting up a new GitHub repository step-by-step.

## Overview

The `setup_repo.sh` script only prepares your **local** repository. You still need to:
1. Create the repository on GitHub (via web interface)
2. Connect your local repo to GitHub
3. Authenticate and push

## Step-by-Step Instructions

### Step 1: Prepare Local Repository

Run the setup script (or do it manually):

```bash
./setup_repo.sh
```

This will:
- Initialize git (if not already done)
- Stage all essential files
- **NOT** push anything to GitHub

### Step 2: Create Initial Commit

```bash
# Review what will be committed
git status

# Create the commit
git commit -m "Initial commit: Multi-agent system with SQL, RAG, and API tools"
```

### Step 3: Create Repository on GitHub

**You must do this via web browser:**

1. Go to https://github.com/new
2. Repository name: Choose a name (e.g., `multiagent-system`)
3. Description: Optional (e.g., "LangGraph-based multi-agent system")
4. Visibility: Public or Private (your choice)
5. **IMPORTANT:** Do NOT check these boxes:
   - ❌ "Add a README file" (you already have one)
   - ❌ "Add .gitignore" (you already have one)
   - ❌ "Choose a license" (optional, but not needed)
6. Click "Create repository"

### Step 4: Connect Local Repo to GitHub

After creating the repo, GitHub will show you instructions. Use the "push an existing repository" option:

```bash
# Replace YOUR_USERNAME and YOUR_REPO_NAME with your actual values
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git branch -M main
git push -u origin main
```

### Step 5: Authenticate When Pushing

When you run `git push`, you'll be prompted for authentication. Choose one method:

#### Option A: GitHub CLI (Easiest)

```bash
# Install GitHub CLI if not installed
# macOS: brew install gh
# Or download from: https://cli.github.com/

# Authenticate
gh auth login

# Now push (won't prompt for credentials)
git push -u origin main
```

#### Option B: Personal Access Token

1. **Create a token:**
   - Go to: https://github.com/settings/tokens
   - Click "Generate new token (classic)"
   - Give it a name (e.g., "multiagent-repo")
   - Select scope: `repo` (this gives full repository access)
   - Click "Generate token"
   - **Copy the token immediately** (you won't see it again)

2. **Use the token:**
   ```bash
   git push -u origin main
   ```
   - Username: Your GitHub username
   - Password: Paste the token (not your GitHub password)

#### Option C: SSH Key

If you already have SSH keys set up with GitHub:

```bash
# Use SSH URL instead
git remote set-url origin git@github.com:YOUR_USERNAME/YOUR_REPO_NAME.git
git push -u origin main
```

## Verification

After pushing, visit your repository on GitHub:
```
https://github.com/YOUR_USERNAME/YOUR_REPO_NAME
```

You should see all your files there.

## Troubleshooting

**"Repository not found" error:**
- Check the repository URL is correct
- Make sure the repository exists on GitHub
- Verify you have access to the repository

**"Authentication failed" error:**
- If using token: Make sure you copied the full token
- If using SSH: Verify your SSH key is added to GitHub
- Try GitHub CLI: `gh auth login`

**"Permission denied" error:**
- Check you have write access to the repository
- Verify your token has `repo` scope
- Make sure you're using the correct GitHub username

## Summary

The `setup_repo.sh` script only handles the **local** setup. The actual GitHub setup is done via:
1. Web browser (creating the repo)
2. Command line (connecting and pushing)
3. Authentication (token, SSH, or GitHub CLI)

No credentials are stored in the script - authentication happens when you push.

