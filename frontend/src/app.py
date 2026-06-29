from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import httpx
import os

app = FastAPI(title="Route Resilience Web Portal", version="1.0.0")

# Backend integration URL
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
print(f"[INFO] Frontend proxy BACKEND_URL set to: {BACKEND_URL}", flush=True)

# Proxy post endpoints to backend
@app.post("/api/infer")
async def proxy_infer(request: Request):
    """
    Proxies topological extraction queries directly to the backend.
    """
    body = await request.body()
    headers = {k: v for k, v in request.headers.items() if k.lower() not in [
        "host", "content-length", "connection", "keep-alive", "proxy-connection", "transfer-encoding"
    ]}
    
    async with httpx.AsyncClient(timeout=300.0) as client:
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
    headers = {k: v for k, v in request.headers.items() if k.lower() not in [
        "host", "content-length", "connection", "keep-alive", "proxy-connection", "transfer-encoding"
    ]}
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(f"{BACKEND_URL}/api/resilience", content=body, headers=headers)
            return resp.json()
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"Resilience analytics gateway error: {e}")

@app.get("/api/presets")
async def proxy_presets():
    """
    Proxies dynamic preset directory mapping queries to the backend.
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(f"{BACKEND_URL}/api/presets")
            return resp.json()
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"Presets listing gateway error: {e}")

@app.post("/api/presets/infer")
async def proxy_presets_infer(request: Request):
    """
    Proxies preset live inference execution queries directly to the backend.
    """
    body = await request.body()
    headers = {k: v for k, v in request.headers.items() if k.lower() not in [
        "host", "content-length", "connection", "keep-alive", "proxy-connection", "transfer-encoding"
    ]}
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            resp = await client.post(f"{BACKEND_URL}/api/presets/infer", content=body, headers=headers)
            return resp.json()
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"Preset inference gateway error: {e}")

@app.post("/api/route/plan")
async def proxy_route_plan(request: Request):
    """
    Proxies dynamic point-to-point shortest path calculations to the backend.
    """
    body = await request.body()
    headers = {k: v for k, v in request.headers.items() if k.lower() not in [
        "host", "content-length", "connection", "keep-alive", "proxy-connection", "transfer-encoding"
    ]}
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(f"{BACKEND_URL}/api/route/plan", content=body, headers=headers)
            return resp.json()
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"Route planning gateway error: {e}")

# Mount static web directory
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
else:
    @app.get("/")
    def read_root():
        return HTMLResponse("<h3>Static directory mounting in progress...</h3>")
