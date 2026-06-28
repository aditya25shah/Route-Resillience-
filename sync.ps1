# Route Resilience System Synchronization Script (PowerShell)

Write-Host "Initializing Git tracking sequence..." -ForegroundColor Cyan
git init

# Set default branch
git branch -M main

Write-Host "Staging decoupled workspace files..." -ForegroundColor Cyan
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
git add sync.sh
git add sync.ps1

Write-Host "Committing workspace structure..." -ForegroundColor Cyan
git commit -m "feat: implement synchronous multi-container orchestration and real-time inference topology pipeline"

Write-Host "Pushing changes to GitHub repository..." -ForegroundColor Cyan
git push -u origin main

Write-Host "Sync operation completed successfully." -ForegroundColor Green
