# Route Resilience System (Baseline: Sat2Graph)

The **Route Resilience System** is an advanced network resilience and routing extraction platform built upon the state-of-the-art **Sat2Graph** model (ECCV 2020) and modernized with **PyTorch 2.x** and **NetworkX**. By converting raw satellite imagery directly into topological road graph networks on the fly, the system enables deep dynamic analysis of route redundancy, network vulnerability, and geographical road resilience.

This version features a **100% dynamic, backend-calculated real-time analytics engine** with a minimalist monochromatic dashboard.

---

## Quick Start & Local Endpoints

After launching the Docker environment, you can access the application layers directly:
*   **Web Dashboard Portal:** [http://localhost:8080](http://localhost:8080)
*   **Analytics REST API:** [http://localhost:8000](http://localhost:8000)
*   **API Interactive Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Architecture & Directory Layout

The project structure is organized into decoupled services:

```text
route-resilience/
│
├── backend/
│   ├── src/
│   │   ├── main.py               # REST API entrypoint (FastAPI)
│   │   ├── inference.py          # Real-time GTE Tensor Mask & skeletonization pipeline
│   │   └── graph_engine.py       # Live NetworkX topology, routing & resilience analytics
│   ├── Requirements.txt          # Modern PyTorch/Skimage dependency stack
│   ├── .dockerignore             # Excludes weights/caches from build context transfer
│   └── Dockerfile                # GPU/CPU PyTorch backend image
│
├── frontend/
│   ├── src/
│   │   ├── app.py                # Static file gateway and API proxy (FastAPI)
│   │   └── static/               # Minimalist monochromatic HTML5 Canvas UI
│   │       ├── index.html
│   │       ├── style.css
│   │       └── app.js
│   ├── Requirements.txt          # Frontend serving dependencies
│   ├── .dockerignore             # Excludes local assets from build context transfer
│   └── Dockerfile                # Frontend container with sync startup check
│
├── data/                         # Host folder containing weights and raster tiles
├── docker-compose.yml            # System orchestrator
└── README.md                     # Technical deployment documentation
```

---

## Deployment & Execution

### Prerequisites
*   Docker & Docker Compose installed.
*   Nvidia Container Toolkit (for optional GPU acceleration).

### 1. Download Model Checkpoints
Since model checkpoints are large (over 2 GB) and excluded from version control, place them in the host `./data/` folder:
1.  Download the model checkpoints folder from the **[Google Drive Link](https://drive.google.com/drive/folders/1FlMcO3Jr8W4qboZUwxgRn6AlYc-AuxQ2?usp=sharing)**.
2.  Extract the contents into `./data/`. The paths should map to `./data/20citiesModel/model.ckpt`.

*Note: The backend has a self-healing morphological fallback. If weights are not found, it runs image skeletonization to extract nodes and edges dynamically, maintaining full application functionality without mockup statistics.*

### 2. Startup Orchestration
Spin up the backend and frontend services using Docker Compose:

```bash
# Build and start all services
docker compose up --build -d
```

During startup:
*   `route-backend` will initialize, loading the model into GPU memory.
*   `route-frontend` will execute a curl check loop (`http://route-backend:8000/health`) and block container boot until the backend is fully initialized and operational.

---

## Endpoints

*   **Frontend Dashboard:** [http://localhost:8080](http://localhost:8080)
*   **Backend REST API:** [http://localhost:8000](http://localhost:8000)
*   **Backend OpenAPI Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Interactive Features

Once inside the minimalist monochromatic UI:
1.  **Select Preset or Upload Imagery:** Choose a preloaded city or drag-and-drop a GeoTIFF tile. The raw image is sent directly to `route-backend` to run GTE extraction and topology vectorization.
2.  **Toggle Display Layers:** Switch the view between raw **Satellite**, **GTE Tensor Mask**, and the **Extracted Graph**.
3.  **Road Disruption Simulator:** Enable "Block Road Node" and click on map intersections. The system instantly recalculates APLS scores and components connectedness in the NetworkX backend.
4.  **Route Planner:** Select start and destination points to draw the shortest path. If a block divides the route, the system attempts to find alternative paths or reports a network partition.
