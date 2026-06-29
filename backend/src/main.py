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
        nodes, edges = pipeline.run_inference(image)
        
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
