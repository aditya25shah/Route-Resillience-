#!/usr/bin/env bash
# Route Resilience System Synchronization Script

echo "Initializing Git tracking sequence..."
git init

# Configure tracking branches if needed
git branch -M main

# Stage all files in backend, frontend, workflows, and root docker files
echo "Staging decoupled workspace files..."
git add .github/workflows/ci-cd.yml
git add backend/Dockerfile
git add backend/Requirements.txt
git add backend/src/main.py
git add backend/src/inference.py
git add backend/src/graph_engine.py
git add frontend/Dockerfile
git add frontend/Requirements.txt
git add frontend/src/app.py
git add frontend/src/static/index.html
git add frontend/src/static/style.css
git add frontend/src/static/app.js
git add docker-compose.yml
git add README.md
git add .gitignore

# Perform atomic commit
echo "Committing workspace structure..."
git commit -m "feat: implement synchronous multi-container orchestration and real-time inference topology pipeline"

# Push to the remote repository
echo "Pushing changes to GitHub repository..."
git push -u origin main

echo "Sync operation completed successfully."
