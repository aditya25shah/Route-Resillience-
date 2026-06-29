import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from skimage.morphology import skeletonize
import networkx as nx

class SimpleGTEModel(nn.Module):
    """
    Minimal representation of Sat2Graph ECCV 2020 neural model structure
    for loading spatial tensor weights.
    """
    def __init__(self):
        super(SimpleGTEModel, self).__init__()
        # Encoder layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # Decoder layers (Graph Tensor Encoding outputs)
        self.dec_edges = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.dec_nodes = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = torch.relu(self.conv2(x))
        edges_mask = torch.sigmoid(self.dec_edges(x))
        nodes_mask = torch.sigmoid(self.dec_nodes(edges_mask))
        return edges_mask, nodes_mask

class InferencePipeline:
    def __init__(self, weights_path="backend/weights/sat2graph_weights.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.weights_path = weights_path
        
        # Self-healing attempt to load neural network weights
        if os.path.exists(self.weights_path):
            try:
                self.model = SimpleGTEModel().to(self.device)
                # Load state dict safely
                self.model.load_state_dict(torch.load(self.weights_path, map_location=self.device), strict=False)
                
                # Apply PyTorch 2.x compilation patterns
                if hasattr(torch, "compile"):
                    try:
                        self.model = torch.compile(self.model)
                        print("[INFO] PyTorch 2.x native compilation successfully applied.")
                    except Exception as compile_err:
                        print(f"[WARNING] torch.compile is not supported in this runtime: {compile_err}")
                
                self.model.eval()
                print(f"[INFO] Sat2Graph neural model successfully loaded on {self.device}")
            except Exception as e:
                print(f"[WARNING] Neural model state load failed: {e}. Falling back to dynamic structural skeletonization.")
                self.model = None
        else:
            print(f"[INFO] Weights file not found at {weights_path}. Running real-time structural skeletonization pipeline.")

    def run_inference(self, image):
        """
        Runs the full inference pipeline:
        1. Resizes input and converts to numpy
        2. Queries model or executesfallback structural filters
        3. Decodes vertices and segments using strict mathematical thresholding and coordinates deduplication.
        """
        img_np = np.array(image)
        if len(img_np.shape) == 2:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        elif img_np.shape[2] == 4:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
            
        h, w = img_np.shape[:2]
        target_w, target_h = 800, 600
        img_resized = cv2.resize(img_np, (target_w, target_h))
        
        # Check if neural model is available
        if self.model is not None:
            try:
                tensor_in = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0).to(self.device) / 255.0
                with torch.no_grad():
                    edges_mask, nodes_mask = self.model(tensor_in)
                    
                # Apply Edgeness Probability Threshold (threshold_edge = 0.45)
                edges_mean = edges_mask.squeeze(0).mean(dim=0)
                edges_binary = (edges_mean >= 0.45).float() * 255.0
                mask_np = edges_binary.cpu().numpy().astype(np.uint8)
                mask_resized = cv2.resize(mask_np, (target_w, target_h))
                
                # Rescale nodes_mask to CPU numpy array to pass as vertexness probability map
                nodes_mean = nodes_mask.squeeze(0).mean(dim=0).cpu().numpy()
                nodes_resized = cv2.resize(nodes_mean, (target_w, target_h))
                
                return self._extract_graph_from_mask(img_resized, mask_resized, vertex_prob_map=nodes_resized)
            except Exception as e:
                print(f"[ERROR] Live PyTorch inference failure: {e}. Defaulting to dynamic structural mode.")
                
        # 2. Dynamic structural skeletonization fallback pipeline
        gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
        smoothed = cv2.bilateralFilter(gray, 9, 75, 75)
        thresh = cv2.adaptiveThreshold(smoothed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 8)
        
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed_mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel)
        open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        clean_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, open_kernel)
        
        return self._extract_graph_from_mask(img_resized, clean_mask, vertex_prob_map=None)

    def _extract_graph_from_mask(self, original_img, binary_mask, vertex_prob_map=None):
        """
        Parses binary GTE/structural segmentation masks to extract nodes and edges.
        """
        bool_mask = binary_mask > 127
        skeleton = skeletonize(bool_mask).astype(np.uint8) * 255
        
        # Find junction points using neighbor count convolution
        kernel = np.array([[1, 1, 1],
                           [1, 0, 1],
                           [1, 1, 1]], dtype=np.uint8)
        
        skel_float = (skeleton > 0).astype(float)
        neighbor_count = cv2.filter2D(skel_float, -1, kernel) * skel_float
        
        junction_y, junction_x = np.where(neighbor_count >= 3)
        term_y, term_x = np.where(neighbor_count == 1)
        
        nodes_raw = list(zip(junction_x, junction_y)) + list(zip(term_x, term_y))
        
        # --- SPATIAL DEDUPLICATION: Grid-snap nodes within 15px radius ---
        # Raw skeleton generates thousands of pixel-level junction points.
        # Collapse nearby points into single representative nodes to keep
        # the working set under ~1000 and prevent KDTree pair explosion.
        nodes = []
        node_id = 1
        node_lookup = {}
        
        for pt in nodes_raw:
            x, y = int(pt[0]), int(pt[1])
            duplicate = False
            for n in nodes:
                if abs(n["x"] - x) < 8 and abs(n["y"] - y) < 8:
                    duplicate = True
                    node_lookup[(x, y)] = n["id"]
                    break
            if not duplicate:
                new_node = {
                    "id": node_id,
                    "x": x,
                    "y": y,
                    "name": f"Junction {node_id}"
                }
                nodes.append(new_node)
                node_lookup[(x, y)] = node_id
                node_id += 1
            
        # --- BOUNDED EDGE LINKING: MAX_RADIUS = 60px (300m at 5m/px) ---
        MAX_RADIUS = 60.0
        edges = []
        if len(nodes) > 1:
            from scipy.spatial import KDTree
            coords = np.array([[n["x"], n["y"]] for n in nodes], dtype=float)
            tree = KDTree(coords)
            pairs = tree.query_pairs(r=MAX_RADIUS)
            
            # Dilate skeleton to allow 2px tolerance on line tracing
            dilated_skel = cv2.dilate(skeleton, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
            
            # Vectorized line trace check for candidate pairs
            steps = 10
            s_vals = np.linspace(0.1, 0.9, steps)
            
            for i, j in pairs:
                n1 = nodes[i]
                n2 = nodes[j]
                
                pxs = (n1["x"] + (n2["x"] - n1["x"]) * s_vals).astype(int)
                pys = (n1["y"] + (n2["y"] - n1["y"]) * s_vals).astype(int)
                
                pxs = np.clip(pxs, 0, skeleton.shape[1] - 1)
                pys = np.clip(pys, 0, skeleton.shape[0] - 1)
                
                hits = np.sum(dilated_skel[pys, pxs] > 0)
                if (hits / steps) > 0.70:
                    edges.append({
                        "from": n1["id"],
                        "to": n2["id"]
                    })
                
        # Delegate post-processing cleanups to GraphEngine
        from src.graph_engine import GraphEngine
        return GraphEngine.clean_graph_topology(nodes, edges, vertex_prob_map)

import math
