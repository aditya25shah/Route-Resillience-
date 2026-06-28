from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import httpx
import os

app = FastAPI(title="Route Resilience Web Portal", version="1.0.0")

# Backend integration URL
BACKEND_URL = os.getenv("BACKEND_URL", "http://route-backend:8000")

# Proxy post endpoints to backend
@app.post("/api/infer")
async def proxy_infer(request: Request):
    """
    Proxies topological extraction queries directly to the backend.
    """
    body = await request.body()
    headers = dict(request.headers)
    # Remove Host header to avoid routing loops
    headers.pop("host", None)
    headers.pop("content-length", None)
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await client.post(f"{BACKEND_URL}/api/infer", content=body, headers=headers)
            return resp.json()
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"Inference gateway error: {e}")

@app.post("/api/resilience")
async def proxy_resilience(request: Request):
    """
    Proxies network degradation and routing query calculations to the backend.
    """
    body = await request.body()
    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(f"{BACKEND_URL}/api/resilience", content=body, headers=headers)
            return resp.json()
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"Resilience analytics gateway error: {e}")

# Mount static web directory
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
else:
    @app.get("/")
    def read_root():
        return HTMLResponse("<h3>Static directory mounting in progress...</h3>")
