from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import base64
from io import BytesIO
from PIL import Image
import os

from src.inference import InferencePipeline
from src.graph_engine import GraphEngine

app = FastAPI(title="Route Resilience Analytics Backend", version="1.0.0")

# CORS middleware for communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline
weights_path = os.getenv("WEIGHTS_PATH", "backend/weights/sat2graph_weights.pth")
pipeline = InferencePipeline(weights_path=weights_path)

class InferenceRequest(BaseModel):
    image: str # Base64 encoded image string
    is_sar: Optional[bool] = False
    cloud_cover: Optional[bool] = False

class CoordinateNode(BaseModel):
    id: int
    x: float
    y: float
    name: Optional[str] = ""

class LinkEdge(BaseModel):
    from_node: int = Field(..., alias="from")
    to_node: int = Field(..., alias="to")

class ResilienceRequest(BaseModel):
    nodes: List[dict]
    edges: List[dict]
    blocked: List[int]
    startNode: Optional[int] = None
    endNode: Optional[int] = None

@app.get("/health")
def health_check():
    """
    Inference service health probe.
    """
    import torch
    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        "weights_loaded": pipeline.model is not None
    }

@app.post("/api/infer")
def perform_inference(req: InferenceRequest):
    """
    Decodes input image, runs GTE mask extraction and vectorizes road topologies.
    """
    try:
        # Parse base64 image data
        header, encoded = req.image.split(",", 1) if "," in req.image else ("", req.image)
        img_bytes = base64.b64decode(encoded)
        image = Image.open(BytesIO(img_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {e}")
        
    try:
        # Run inference and topology extraction
        nodes, edges = pipeline.run_inference(image, is_sar=req.is_sar, cloud_cover=req.cloud_cover)
        
        # Clean graph topology
        nodes, edges = GraphEngine.clean_graph_topology(nodes, edges)
        
        # Calculate base metrics
        metrics = GraphEngine.compute_resilience_metrics(nodes, edges)
        
        return {
            "success": True,
            "nodes": nodes,
            "edges": edges,
            "metrics": metrics
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference pipeline execution error: {e}")

@app.post("/api/resilience")
def calculate_resilience(req: ResilienceRequest):
    """
    Recalculates APLS and network component connectedness based on blocked node lists.
    """
    try:
        # Compute dynamic metrics using GraphEngine
        metrics = GraphEngine.compute_resilience_metrics(req.nodes, req.edges, req.blocked)
        
        # Calculate shortest path if requested
        shortest_path = None
        if req.startNode is not None and req.endNode is not None:
            shortest_path = GraphEngine.get_shortest_path(
                req.nodes, req.edges, req.startNode, req.endNode, req.blocked
            )
            
        return {
            "success": True,
            "metrics": metrics,
            "shortest_path": shortest_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Resilience calculation error: {e}")



class RoutePlanRequest(BaseModel):
    origin_node_id: int
    destination_node_id: int
    nodes: List[dict]
    edges: List[dict]
    blocked: List[int]

@app.post("/api/route/plan")
def route_plan(req: RoutePlanRequest):
    """
    Computes shortest path between origin and destination on the NetworkX graph.
    Returns path nodes list, total distance in meters, and complexity.
    """
    try:
        # Build NetworkX graph from nodes/edges, omitting blocked ones
        G = GraphEngine.build_networkx_graph(req.nodes, req.edges, req.blocked)
        
        origin = int(req.origin_node_id)
        destination = int(req.destination_node_id)
        
        if not G.has_node(origin) or not G.has_node(destination):
            raise HTTPException(status_code=404, detail="PathNotFound")
            
        # Run A* Search with Euclidean distance metric heuristic
        def heuristic(u, v):
            nu = G.nodes[u]
            nv = G.nodes[v]
            return math.hypot(nu['x'] - nv['x'], nu['y'] - nv['y'])
            
        try:
            path = nx.astar_path(G, origin, destination, heuristic=heuristic, weight='weight')
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            raise HTTPException(status_code=404, detail="PathNotFound")
            
        # Calculate total distance in meters (weight is pixel distance, 1px = 5m)
        total_dist_pixels = sum(G[path[i]][path[i+1]]['weight'] for i in range(len(path)-1))
        total_dist_meters = int(round(total_dist_pixels * 5.0))
        
        return {
            "success": True,
            "path": path,
            "distance_meters": total_dist_meters,
            "complexity": len(path) - 1
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Route planning error: {e}")

@app.get("/api/diagnostics")
def system_diagnostics():
    """
    Runs self-testing diagnostic scans on the weights file, model load status,
    and performs a synthetic mock inference iteration. Returns a comprehensive
    system health and troubleshooting report.
    """
    import time
    import hashlib
    import traceback
    
    report = {
        "timestamp": time.time(),
        "status": "OK",
        "checks": {},
        "errors": []
    }
    
    # 1. Check Weights file
    t0 = time.time()
    w_path = os.getenv("WEIGHTS_PATH", "backend/weights/sat2graph_weights.pth")
    report["checks"]["weights_path_env"] = w_path
    
    # Resolve actual file path
    possible_paths = [
        w_path,
        "backend/weights/sat2graph_weights.pth",
        "../backend/weights/sat2graph_weights.pth",
        "data/20citiesModel/model.ckpt",
        "../data/20citiesModel/model.ckpt"
    ]
    resolved_path = None
    for p in possible_paths:
        if p and os.path.exists(p):
            resolved_path = p
            break
            
    if resolved_path:
        report["checks"]["weights_file_exists"] = True
        report["checks"]["weights_file_resolved"] = resolved_path
        size_bytes = os.path.getsize(resolved_path)
        report["checks"]["weights_file_size_bytes"] = size_bytes
        
        # Check checksum signature
        try:
            with open(resolved_path, "rb") as f:
                checksum = hashlib.sha256(f.read()).hexdigest()
            report["checks"]["weights_file_sha256"] = checksum
            expected_hash = "bfa5500962446da0353bbea9129d089361ca025c6e3c5c8c26516b3b3bf14525"
            if checksum == expected_hash:
                report["checks"]["weights_integrity"] = "VALID"
            else:
                report["checks"]["weights_integrity"] = "CORRUPTED_SIGNATURE_MISMATCH"
                report["errors"].append("Weights checksum does not match expected signature.")
                report["status"] = "DEGRADED"
        except Exception as hash_err:
            report["checks"]["weights_integrity"] = "READ_ERROR"
            report["errors"].append(f"Weights integrity check failed: {hash_err}")
            report["status"] = "DEGRADED"
    else:
        report["checks"]["weights_file_exists"] = False
        report["errors"].append("Weights file not found in any standard path.")
        report["status"] = "ERROR"
        
    report["checks"]["weights_check_elapsed_sec"] = time.time() - t0
    
    # 2. Check pipeline initialization
    report["checks"]["pipeline_initialized"] = (pipeline is not None)
    if pipeline:
        report["checks"]["pipeline_device"] = str(pipeline.device)
        report["checks"]["pipeline_model_loaded"] = (pipeline.model is not None)
        
    # 3. Perform synthetic end-to-end trace mock run
    if pipeline and pipeline.model is not None:
        try:
            t_test = time.time()
            # Generate a simple synthetic 100x100 RGB image with a cross grid
            import numpy as np
            from PIL import Image
            grid_img = np.zeros((100, 100, 3), dtype=np.uint8)
            grid_img[50, :, :] = 255
            grid_img[:, 50, :] = 255
            pil_test = Image.fromarray(grid_img)
            
            # Execute inference
            raw_nodes, raw_edges = pipeline.run_inference(pil_test)
            clean_nodes, clean_edges = GraphEngine.clean_graph_topology(raw_nodes, raw_edges)
            metrics = GraphEngine.compute_resilience_metrics(clean_nodes, clean_edges)
            
            report["checks"]["synthetic_inference"] = {
                "success": True,
                "elapsed_sec": time.time() - t_test,
                "raw_nodes": len(raw_nodes),
                "raw_edges": len(raw_edges),
                "clean_nodes": len(clean_nodes),
                "clean_edges": len(clean_edges),
                "metrics": metrics
            }
        except Exception as test_err:
            report["status"] = "ERROR"
            report["checks"]["synthetic_inference"] = {
                "success": False,
                "error": str(test_err)
            }
            report["errors"].append({
                "stage": "Synthetic inference pipeline execution",
                "message": str(test_err),
                "traceback": traceback.format_exc()
            })
            
    return report
