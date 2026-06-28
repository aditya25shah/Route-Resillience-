FROM python:3.10-slim

# Install system utilities including curl for health probing
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install python dependencies
COPY Requirements.txt .
RUN pip install --no-cache-dir -r Requirements.txt

# Copy application sources
COPY src/ /app/src/

# Expose port 8080
EXPOSE 8080

# Sync-wait loop command inside default shell entrypoint
CMD sh -c "echo 'Probing route-backend service availability...' && \
           until curl -s http://route-backend:8000/health | grep -q 'healthy'; do \
               echo 'Waiting for backend container to load neural network weights...'; \
               sleep 2; \
           done && \
           echo 'Backend synchronised. Starting frontend...' && \
           uvicorn src.app:app --host 0.0.0.0 --port 8080"
