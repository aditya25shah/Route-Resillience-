// Global App Viewport & Pan-Zoom State
let zoomScale = 1.0;
let panX = 0.0;
let panY = 0.0;
let isPanning = false;
let startPanX = 0;
let startPanY = 0;
let hasPanned = false;
let mouseDownX = 0;
let mouseDownY = 0;

// Global App State
let blockedNodes = new Set();
let currentMode = "inspect"; // inspect, block, route
let routePoints = []; // Source and Target nodes for Route Planner
let activeShortestPath = null; // Calculated by backend
let activeRouteMeta = null; // Shortest path distance/complexity metadata
let hoverNode = null;
let hoverEdge = null; // Tracked for dynamic inline tooltips
let customImageLoaded = false;
let customImageSrc = null;

// Graph Model structure
let activeGraph = { nodes: [], edges: [], stats: {}, apls: 0.0 };

// Canvas Elements
const canvas = document.getElementById("mapCanvas");
const ctx = canvas.getContext("2d");
const wrapper = document.getElementById("canvasWrapper");

// UI Elements
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const loadingOverlay = document.getElementById("loadingOverlay");
const progressBarFill = document.getElementById("progressBarFill");
const loadingStep = document.getElementById("loadingStep");
const systemStatus = document.getElementById("systemStatus");

const resilienceGauge = document.getElementById("resilienceGauge");
const valResilience = document.getElementById("valResilience");
const valAPLS = document.getElementById("valAPLS");
const aplsFill = document.getElementById("aplsFill");

const statEdges = document.getElementById("statEdges");
const statNodes = document.getElementById("statNodes");
const statAvgLen = document.getElementById("statAvgLen");
const statRedundancy = document.getElementById("statRedundancy");

const chkSatellite = document.getElementById("chkSatellite");
const chkMask = document.getElementById("chkMask");
const chkGraph = document.getElementById("chkGraph");

const btnInspect = document.getElementById("btnInspect");
const btnBlock = document.getElementById("btnBlock");
const btnRoute = document.getElementById("btnRoute");
const btnClearBlocks = document.getElementById("btnClearBlocks");




// Set up event listeners
window.addEventListener("resize", resizeCanvas);
if (chkSatellite) chkSatellite.addEventListener("change", render);
if (chkMask) chkMask.addEventListener("change", render);
if (chkGraph) chkGraph.addEventListener("change", render);

if (btnInspect) btnInspect.addEventListener("click", () => setMode("inspect"));
if (btnBlock) btnBlock.addEventListener("click", () => setMode("block"));
if (btnRoute) btnRoute.addEventListener("click", () => setMode("route"));
if (btnClearBlocks) btnClearBlocks.addEventListener("click", clearBlocks);

// Setup dropzone
if (dropzone) {
    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.classList.add("dragover");
    });

    dropzone.addEventListener("dragleave", () => {
        dropzone.classList.remove("dragover");
    });

    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });
}

if (fileInput) {
    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });
}

// Canvas Interaction Mouse Event Listeners
canvas.addEventListener("mousemove", onMouseMove);
canvas.addEventListener("click", onMouseClick);
canvas.addEventListener("mouseleave", () => {
    hoverNode = null;
    hoverEdge = null;
    render();
});

// Panning Mouse Event Listeners
canvas.addEventListener("mousedown", (e) => {
    if (e.button === 0) { // Left-click drag pans empty space
        isPanning = true;
        startPanX = e.clientX - panX;
        startPanY = e.clientY - panY;
        mouseDownX = e.clientX;
        mouseDownY = e.clientY;
        hasPanned = false;
    }
});

window.addEventListener("mouseup", (e) => {
    if (isPanning) {
        isPanning = false;
        if (Math.hypot(e.clientX - mouseDownX, e.clientY - mouseDownY) > 5) {
            hasPanned = true;
        }
    }
});

// Wheel zoom event listener
canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    const zoomFactor = 1.1;
    const oldScale = zoomScale;
    
    if (e.deltaY < 0) {
        zoomScale = Math.min(zoomScale * zoomFactor, 15.0); // max zoom 15x
    } else {
        zoomScale = Math.max(zoomScale / zoomFactor, 0.5); // min zoom 0.5x
    }

    // Centered zoom correction
    panX = mouseX - (mouseX - panX) * (zoomScale / oldScale);
    panY = mouseY - (mouseY - panY) * (zoomScale / oldScale);

    render();
});

// Initialize App
resizeCanvas();

// Function definitions
function resizeCanvas() {
    const width = wrapper.clientWidth;
    const height = wrapper.clientHeight;
    
    canvas.width = width;
    canvas.height = height;
    
    render();
}



function updateStatsDisplay() {
    if (valAPLS) valAPLS.textContent = Number(activeGraph.apls).toFixed(3);
    if (aplsFill) aplsFill.style.width = (activeGraph.apls * 100) + "%";
    
    if (statEdges) statEdges.textContent = activeGraph.stats.edges;
    if (statNodes) statNodes.textContent = activeGraph.stats.nodes;
    if (statAvgLen) statAvgLen.textContent = activeGraph.stats.avgLen;
    if (statRedundancy) statRedundancy.textContent = activeGraph.stats.redundancy;
}

function handleFile(file) {
    customImageLoaded = true;
    const reader = new FileReader();
    reader.onload = function(event) {
        triggerInferencePipeline(event.target.result);
    };
    reader.readAsDataURL(file);
}

function triggerInferencePipeline(imgDataSrc) {
    let isLoading = true;
    if (loadingOverlay) loadingOverlay.classList.add("active");
    if (progressBarFill) progressBarFill.style.width = "10%";
    if (loadingStep) {
        loadingStep.style.color = "";
        loadingStep.style.fontFamily = "";
        loadingStep.textContent = "Stage 1: READING RASTER GEOMETRY...";
    }
    if (systemStatus) systemStatus.textContent = "Running live inference...";
    
    // Fine-line radar spinner
    const spinner = document.querySelector(".spinner");
    if (spinner) spinner.style.display = "block";
    
    // Simulate pipeline stages using timeouts while request runs in background
    let stageTimer = setTimeout(() => {
        if (isLoading && loadingStep) {
            loadingStep.textContent = "Stage 2: EXECUTING SAT2GRAPH INFERENCE...";
            if (progressBarFill) progressBarFill.style.width = "40%";
        }
    }, 1000);
    
    let stageTimer2 = setTimeout(() => {
        if (isLoading && loadingStep) {
            loadingStep.textContent = "Stage 3: APPLYING TOPOLOGICAL FILTERS...";
            if (progressBarFill) progressBarFill.style.width = "75%";
        }
    }, 2500);
    
    fetch("/api/infer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: imgDataSrc })
    })
    .then(response => {
        isLoading = false;
        clearTimeout(stageTimer);
        clearTimeout(stageTimer2);
        if (!response.ok) throw new Error("Inference execution failed");
        return response.json();
    })
    .then(data => {
        if (progressBarFill) progressBarFill.style.width = "100%";
        if (loadingStep) loadingStep.textContent = "Inference completed successfully.";
        
        setTimeout(() => {
            if (loadingOverlay) loadingOverlay.classList.remove("active");
            if (systemStatus) systemStatus.textContent = "Active - Live Extracted";
            
            activeGraph = {
                stats: {
                    totalLen: data.metrics.total_len_km,
                    nodes: data.metrics.active_nodes,
                    edges: data.metrics.active_edges,
                    avgLen: data.metrics.avg_len_km,
                    redundancy: data.metrics.redundancy,
                    resilience: data.metrics.resilience + "%",
                    apls: Number(data.metrics.apls).toFixed(3)
                },
                apls: data.metrics.apls,
                nodes: data.nodes,
                edges: data.edges
            };
            
            if (customImageLoaded) {
                customImageSrc = new Image();
                customImageSrc.src = imgDataSrc;
                customImageSrc.onload = () => {
                    updateStatsDisplay();
                    updateResilienceUI(data.metrics);
                    resizeCanvas();
                };
            } else {
                updateStatsDisplay();
                updateResilienceUI(data.metrics);
                resizeCanvas();
            }
        }, 300);
    })
    .catch(err => {
        isLoading = false;
        clearTimeout(stageTimer);
        clearTimeout(stageTimer2);
        console.error(err);
        
        // INTERCEPT TIMEOUT / FAILURE: Hide spinner and display clean white error message
        if (spinner) spinner.style.display = "none";
        if (loadingStep) {
            loadingStep.style.color = "#ffffff";
            loadingStep.style.fontFamily = "'SF Mono', monospace";
            loadingStep.textContent = "INFERENCE TIMEOUT // CHECK WEIGHTS PATH";
        }
        if (progressBarFill) {
            progressBarFill.style.width = "100%";
            progressBarFill.style.backgroundColor = "#ff0000";
        }
        if (systemStatus) systemStatus.textContent = "Inference Timeout";
        
        // Let it persist for 6 seconds so user can read it, then reset
        setTimeout(() => {
            if (loadingOverlay) loadingOverlay.classList.remove("active");
            // Reset styles for next attempt
            if (spinner) spinner.style.display = "block";
            if (loadingStep) {
                loadingStep.style.color = "";
                loadingStep.style.fontFamily = "";
            }
            if (progressBarFill) {
                progressBarFill.style.backgroundColor = "";
            }
        }, 6000);
    });
}

function setMode(mode) {
    currentMode = mode;
    
    if (btnInspect) btnInspect.classList.remove("active");
    if (btnBlock) btnBlock.classList.remove("active");
    if (btnRoute) btnRoute.classList.remove("active");
    
    if (mode === "inspect" && btnInspect) btnInspect.classList.add("active");
    if (mode === "block" && btnBlock) btnBlock.classList.add("active");
    if (mode === "route" && btnRoute) btnRoute.classList.add("active");
    
    routePoints = [];
    activeShortestPath = null;
    activeRouteMeta = null;
    showPlaceholderDetail();
    render();
}

function clearBlocks() {
    blockedNodes.clear();
    calculateResilience();
    showPlaceholderDetail();
    render();
}

function calculateResilience() {
    if (activeGraph.nodes.length === 0) return;
    
    const payload = {
        nodes: activeGraph.nodes,
        edges: activeGraph.edges,
        blocked: Array.from(blockedNodes),
        startNode: routePoints[0] !== undefined ? routePoints[0] : null,
        endNode: routePoints[1] !== undefined ? routePoints[1] : null
    };
    
    fetch("/api/resilience", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
    })
    .then(response => {
        if (!response.ok) throw new Error("Resilience request failed");
        return response.json();
    })
    .then(data => {
        if (data.success) {
            updateResilienceUI(data.metrics);
            activeShortestPath = data.shortest_path;
            
            // Sync calculations to local state stats for inline tooltip rendering
            activeGraph.stats.resilience = data.metrics.resilience + "%";
            activeGraph.stats.apls = Number(data.metrics.apls).toFixed(3);
            activeGraph.stats.edges = data.metrics.active_edges;
            activeGraph.stats.nodes = data.metrics.active_nodes;
            activeGraph.stats.avgLen = data.metrics.avg_len_km;
            activeGraph.stats.redundancy = data.metrics.redundancy;
            activeGraph.stats.totalLen = data.metrics.total_len_km;
            
            if (currentMode === "route") {
                showRouteDetail();
                planRoute();
            } else {
                render();
            }
        }
    })
    .catch(err => console.error("Error fetching resilience metrics:", err));
}

function planRoute() {
    if (routePoints.length !== 2) {
        activeShortestPath = null;
        activeRouteMeta = null;
        render();
        return;
    }
    
    const payload = {
        origin_node_id: routePoints[0],
        destination_node_id: routePoints[1],
        nodes: activeGraph.nodes,
        edges: activeGraph.edges,
        blocked: Array.from(blockedNodes)
    };
    
    fetch("/api/route/plan", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
    })
    .then(response => {
        if (!response.ok) {
            if (response.status === 404) {
                throw new Error("PathNotFound");
            }
            throw new Error("Route planning failed");
        }
        return response.json();
    })
    .then(data => {
        if (data.success) {
            activeShortestPath = data.path;
            activeRouteMeta = {
                distance: data.distance_meters,
                complexity: data.complexity
            };
            render();
        }
    })
    .catch(err => {
        console.error("Route planning error:", err);
        activeShortestPath = null;
        activeRouteMeta = null;
        render();
    });
}

function updateResilienceUI(metrics) {
    if (valResilience) valResilience.textContent = metrics.resilience + "%";
    
    if (resilienceGauge) {
        // Circumference = 2 * PI * 18 = 113.1
        const offset = 113 * (1 - metrics.resilience / 100);
        resilienceGauge.style.strokeDashoffset = offset;
    }
    
    if (valAPLS) valAPLS.textContent = Number(metrics.apls).toFixed(3);
    if (aplsFill) aplsFill.style.width = (metrics.apls * 100) + "%";
    
    if (statEdges) statEdges.textContent = metrics.active_edges;
    if (statNodes) statNodes.textContent = metrics.active_nodes;
    if (statAvgLen) statAvgLen.textContent = metrics.avg_len_km;
    if (statRedundancy) statRedundancy.textContent = metrics.redundancy;
}

function getDistanceBetweenNodes(n1Id, n2Id) {
    const n1 = activeGraph.nodes.find(n => n.id === n1Id);
    const n2 = activeGraph.nodes.find(n => n.id === n2Id);
    if (!n1 || !n2) return 0;
    return Math.hypot(n1.x - n2.x, n1.y - n2.y);
}

// Helper to check if an edge is part of the computed shortest path
function isEdgeInShortestPath(fromId, toId) {
    if (!activeShortestPath || activeShortestPath.length < 2) return false;
    for (let i = 0; i < activeShortestPath.length - 1; i++) {
        const u = activeShortestPath[i];
        const v = activeShortestPath[i+1];
        if ((u === fromId && v === toId) || (u === toId && v === fromId)) {
            return true;
        }
    }
    return false;
}

// Render loop
function render() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    const showSat = chkSatellite ? chkSatellite.checked : true;
    const showMask = chkMask ? chkMask.checked : true;
    const showGraph = chkGraph ? chkGraph.checked : true;
    
    // 1. Draw Satellite Background
    if (showSat) {
        if (customImageLoaded && customImageSrc) {
            ctx.drawImage(customImageSrc, panX, panY, canvas.width * zoomScale, canvas.height * zoomScale);
        }
    } else {
        // Dark blueprint background
        ctx.fillStyle = "#000000";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Grid pattern (relative to pan and zoom Scale)
        ctx.strokeStyle = "rgba(255, 255, 255, 0.02)";
        ctx.lineWidth = 1;
        const gridSz = 40 * zoomScale;
        const startX = panX % gridSz;
        const startY = panY % gridSz;
        
        for (let x = startX; x < canvas.width; x += gridSz) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, canvas.height);
            ctx.stroke();
        }
        for (let y = startY; y < canvas.height; y += gridSz) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(canvas.width, y);
            ctx.stroke();
        }
    }
    
    // 2. Draw GTE Segmentation Mask
    if (showMask) {
        ctx.save();
        ctx.globalCompositeOperation = showSat ? "screen" : "source-over";
        
        activeGraph.edges.forEach(e => {
            const n1 = activeGraph.nodes.find(n => n.id === e.from);
            const n2 = activeGraph.nodes.find(n => n.id === e.to);
            if (!n1 || !n2) return;
            
            const p1 = getScaledCoords(n1);
            const p2 = getScaledCoords(n2);
            
            ctx.beginPath();
            ctx.moveTo(p1.x, p1.y);
            ctx.lineTo(p2.x, p2.y);
            ctx.strokeStyle = "rgba(255, 255, 255, 0.08)";
            ctx.lineWidth = Math.max(3.0, 14 / Math.sqrt(zoomScale));
            ctx.lineCap = "round";
            ctx.stroke();
            
            ctx.strokeStyle = "rgba(255, 255, 255, 0.15)";
            ctx.lineWidth = Math.max(2.0, 8 / Math.sqrt(zoomScale));
            ctx.stroke();
        });
        ctx.restore();
    }
    
    // 3. Draw Extracted Graph Overlay
    if (showGraph) {
        const edgeWidth = Math.max(0.4, 1.5 / Math.sqrt(zoomScale));
        const routeActive = (currentMode === "route" && routePoints.length === 2 && activeShortestPath);
        
        // Draw Edges
        activeGraph.edges.forEach(e => {
            const n1 = activeGraph.nodes.find(n => n.id === e.from);
            const n2 = activeGraph.nodes.find(n => n.id === e.to);
            if (!n1 || !n2) return;
            
            const p1 = getScaledCoords(n1);
            const p2 = getScaledCoords(n2);
            
            ctx.beginPath();
            ctx.moveTo(p1.x, p1.y);
            ctx.lineTo(p2.x, p2.y);
            
            let isHovered = (hoverEdge && hoverEdge.from === e.from && hoverEdge.to === e.to);
            let onPath = routeActive && isEdgeInShortestPath(e.from, e.to);
            
            if (routeActive) {
                if (onPath) {
                    ctx.strokeStyle = "#ffffff";
                    ctx.lineWidth = edgeWidth * 2.5;
                    ctx.setLineDash([]);
                } else {
                    // Muted background segment: slate-grey (#333333)
                    ctx.strokeStyle = "#333333";
                    ctx.lineWidth = edgeWidth;
                    ctx.setLineDash([]);
                }
            } else {
                if (blockedNodes.has(e.from) || blockedNodes.has(e.to)) {
                    ctx.strokeStyle = "rgba(255, 255, 255, 0.15)";
                    ctx.lineWidth = edgeWidth;
                    ctx.setLineDash([4, 4]);
                } else if (isHovered) {
                    ctx.save();
                    ctx.strokeStyle = "#ffffff";
                    ctx.lineWidth = edgeWidth * 2.5;
                    ctx.shadowColor = "#ffffff";
                    ctx.shadowBlur = Math.max(2.0, 8 / Math.sqrt(zoomScale));
                    ctx.setLineDash([]);
                } else {
                    ctx.strokeStyle = "#ffffff";
                    ctx.lineWidth = edgeWidth;
                    ctx.setLineDash([]);
                }
            }
            
            ctx.stroke();
            if (isHovered && !routeActive) {
                ctx.restore();
            }
        });
        ctx.setLineDash([]);
        
        // Draw thick glowing shortest path overlay line
        if (routeActive) {
            ctx.save();
            ctx.beginPath();
            const startNode = activeGraph.nodes.find(n => n.id === activeShortestPath[0]);
            if (startNode) {
                const pStart = getScaledCoords(startNode);
                ctx.moveTo(pStart.x, pStart.y);
                for (let i = 1; i < activeShortestPath.length; i++) {
                    const node = activeGraph.nodes.find(n => n.id === activeShortestPath[i]);
                    if (node) {
                        const pNode = getScaledCoords(node);
                        ctx.lineTo(pNode.x, pNode.y);
                    }
                }
                ctx.strokeStyle = "rgba(255, 255, 255, 0.4)";
                ctx.lineWidth = Math.max(2.0, 6 / Math.sqrt(zoomScale));
                ctx.shadowColor = "#ffffff";
                ctx.shadowBlur = Math.max(4.0, 12 / Math.sqrt(zoomScale));
                ctx.stroke();
            }
            ctx.restore();
        }
        
        // Draw Nodes
        const innerRadius = Math.max(1.5, 4.0 / Math.sqrt(zoomScale));
        const outerRadius = Math.max(2.5, 7.0 / Math.sqrt(zoomScale));
        const outerLineWidth = Math.max(0.4, 1.0 / Math.sqrt(zoomScale));
        
        activeGraph.nodes.forEach(node => {
            const p = getScaledCoords(node);
            
            ctx.beginPath();
            ctx.arc(p.x, p.y, innerRadius, 0, 2 * Math.PI);
            
            let color = "#ffffff";
            let strokeColor = "rgba(255, 255, 255, 0.2)";
            let glow = false;
            let onPath = routeActive && activeShortestPath.includes(node.id);
            
            if (routeActive) {
                if (onPath || routePoints.includes(node.id)) {
                    if (blockedNodes.has(node.id)) {
                        color = "#000000";
                        strokeColor = "#ffffff";
                    } else if (routePoints.includes(node.id)) {
                        color = "#ffffff";
                        strokeColor = "#ffffff";
                        glow = true;
                    } else {
                        color = "#ffffff";
                        strokeColor = "rgba(255, 255, 255, 0.5)";
                    }
                } else {
                    // Muted background node: slate-grey (#333333)
                    color = "#333333";
                    strokeColor = "rgba(51, 51, 51, 0.5)";
                }
            } else {
                if (blockedNodes.has(node.id)) {
                    color = "#000000";
                    strokeColor = "#ffffff";
                } else if (currentMode === "route" && routePoints.includes(node.id)) {
                    color = "#ffffff";
                    strokeColor = "#ffffff";
                    glow = true;
                } else if (hoverNode && hoverNode.id === node.id) {
                    color = "#ffffff";
                    strokeColor = "#ffffff";
                    glow = true;
                }
            }
            
            ctx.fillStyle = color;
            if (glow) {
                ctx.save();
                ctx.shadowColor = color;
                ctx.shadowBlur = Math.max(2.0, 8 / Math.sqrt(zoomScale));
                ctx.fill();
                ctx.restore();
            } else {
                ctx.fill();
            }
            
            // Outer ring
            ctx.beginPath();
            ctx.arc(p.x, p.y, outerRadius, 0, 2 * Math.PI);
            ctx.strokeStyle = strokeColor;
            ctx.lineWidth = outerLineWidth;
            ctx.stroke();
        });
        
        // --- 4. MINIMALIST HUD METRIC OVERLAY ---
        if (currentMode === "route" && routePoints.length === 2) {
            const nodeB = activeGraph.nodes.find(n => n.id === routePoints[1]);
            if (nodeB) {
                const pB = getScaledCoords(nodeB);
                ctx.save();
                ctx.font = "bold 9px monospace";
                ctx.textAlign = "left";
                ctx.textBaseline = "middle";
                ctx.shadowColor = "#000000";
                ctx.shadowBlur = 4;
                
                if (activeShortestPath && activeRouteMeta) {
                    ctx.fillStyle = "#ffffff";
                    const hudText = `ROUTE ESTABLISHED // DISTANCE: ${activeRouteMeta.distance} METERS // PATH COMPLEXITY: ${activeRouteMeta.complexity}`;
                    ctx.fillText(hudText, pB.x + 18, pB.y);
                } else {
                    ctx.fillStyle = "#ff3333";
                    ctx.fillText("ROUTE BLOCKED // PATH DISCONNECTED // 0 METERS", pB.x + 18, pB.y);
                }
                ctx.restore();
            }
        }
    }
}

// Convert absolute coordinates stored in graph (from original 800x600 coordinate grid) to current viewport coordinates
function getScaledCoords(node) {
    // Presets and generated custom graphs are designed relative to 800x600 aspect coordinates
    const scaleX = canvas.width / 800;
    const scaleY = canvas.height / 600;
    
    const sx = node.x * scaleX;
    const sy = node.y * scaleY;
    
    return {
        x: sx * zoomScale + panX,
        y: sy * zoomScale + panY
    };
}

// Reverse conversion (from screen client x,y to 800x600 absolute grid coords)
function getAbsoluteCoords(clientX, clientY) {
    const rect = canvas.getBoundingClientRect();
    const x_canvas = clientX - rect.left;
    const y_canvas = clientY - rect.top;
    
    // Reverse zoom and pan transformation
    const x_screen = (x_canvas - panX) / zoomScale;
    const y_screen = (y_canvas - panY) / zoomScale;
    
    const scaleX = 800 / rect.width;
    const scaleY = 600 / rect.height;
    return {
        x: x_screen * scaleX,
        y: y_screen * scaleY
    };
}

// Distance from point p to line segment ab
function getDistanceToSegment(p, a, b) {
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const l2 = dx * dx + dy * dy;
    if (l2 === 0) return Math.hypot(p.x - a.x, p.y - a.y);
    
    let t = ((p.x - a.x) * dx + (p.y - a.y) * dy) / l2;
    t = Math.max(0, Math.min(1, t));
    
    const cx = a.x + t * dx;
    const cy = a.y + t * dy;
    
    return Math.hypot(p.x - cx, p.y - cy);
}

// Mouse Event handlers
function onMouseMove(e) {
    if (isPanning) {
        panX = e.clientX - startPanX;
        panY = e.clientY - startPanY;
        render();
        return;
    }
    
    const coords = getAbsoluteCoords(e.clientX, e.clientY);
    
    // 1. Check node hover
    let foundNode = null;
    activeGraph.nodes.forEach(node => {
        const dist = Math.hypot(node.x - coords.x, node.y - coords.y);
        if (dist <= 16) { // slightly wider target hit zone
            foundNode = node;
        }
    });
    
    // 2. Check edge hover (only if no node is hovered)
    let foundEdge = null;
    if (!foundNode && activeGraph.edges.length > 0) {
        let minEdgeDist = 12; // hover threshold in pixels
        activeGraph.edges.forEach(edge => {
            const n1 = activeGraph.nodes.find(n => n.id === edge.from);
            const n2 = activeGraph.nodes.find(n => n.id === edge.to);
            if (n1 && n2) {
                const dist = getDistanceToSegment(coords, n1, n2);
                if (dist < minEdgeDist) {
                    minEdgeDist = dist;
                    foundEdge = edge;
                }
            }
        });
    }
    
    let stateChanged = false;
    if (foundNode !== hoverNode) {
        hoverNode = foundNode;
        stateChanged = true;
    }
    if (foundEdge !== hoverEdge) {
        hoverEdge = foundEdge;
        stateChanged = true;
    }
    
    if (stateChanged) {
        render();
    }
    
    // Display minimal floating tooltip if INSPECT mode is active
    if (currentMode === "inspect") {
        const tooltip = document.getElementById("telemetryTooltip");
        if (foundNode) {
            const connections = activeGraph.edges.filter(e => e.from === foundNode.id || e.to === foundNode.id);
            const degCentrality = connections.length;
            
            tooltip.style.display = "block";
            // Make it look clean/minimal with no background cards, borders or shadows (per prompt)
            tooltip.style.background = "none";
            tooltip.style.border = "none";
            tooltip.style.boxShadow = "none";
            tooltip.style.padding = "0";
            tooltip.style.color = "#ffffff";
            tooltip.style.textShadow = "0px 0px 4px #000000"; // text shadow for readability on imagery
            
            tooltip.style.left = `${e.clientX + 15}px`;
            tooltip.style.top = `${e.clientY + 15}px`;
            
            tooltip.innerHTML = `
                <div style="font-family: monospace; font-size: 11px; line-height: 1.4;">
                    <strong>JUNCTION #${foundNode.id}</strong><br/>
                    Degree Centrality: ${degCentrality}<br/>
                    Coord: ${foundNode.x.toFixed(0)}m, ${foundNode.y.toFixed(0)}m
                </div>
            `;
        } else if (foundEdge) {
            const n1 = activeGraph.nodes.find(n => n.id === foundEdge.from);
            const n2 = activeGraph.nodes.find(n => n.id === foundEdge.to);
            const dx = n1.x - n2.x;
            const dy = n1.y - n2.y;
            const pixelLen = Math.hypot(dx, dy);
            const meterLen = Math.round(pixelLen * 5); // 0.005 km per pixel = 5m
            
            tooltip.style.display = "block";
            tooltip.style.background = "none";
            tooltip.style.border = "none";
            tooltip.style.boxShadow = "none";
            tooltip.style.padding = "0";
            tooltip.style.color = "#ffffff";
            tooltip.style.textShadow = "0px 0px 4px #000000";
            
            tooltip.style.left = `${e.clientX + 15}px`;
            tooltip.style.top = `${e.clientY + 15}px`;
            
            tooltip.innerHTML = `
                <div style="font-family: monospace; font-size: 11px; line-height: 1.4;">
                    <strong>EDGE #${foundEdge.from}-${foundEdge.to}</strong><br/>
                    Edge Length: ${meterLen}m
                </div>
            `;
        } else {
            tooltip.style.display = "none";
        }
    }
}

function onMouseClick(e) {
    if (hasPanned) {
        hasPanned = false;
        return;
    }
    
    if (!hoverNode) {
        // If clicking empty canvas area, close tooltip
        document.getElementById("telemetryTooltip").style.display = "none";
        if (currentMode === "route") {
            routePoints = [];
            activeShortestPath = null;
        }
        render();
        return;
    }
    
    const node = hoverNode;
    
    if (currentMode === "inspect") {
        showNodeDetail(node);
    } else if (currentMode === "block") {
        if (blockedNodes.has(node.id)) {
            blockedNodes.delete(node.id);
        } else {
            blockedNodes.add(node.id);
        }
        calculateResilience();
        showNodeDetail(node);
        render();
    } else if (currentMode === "route") {
        if (routePoints.includes(node.id)) {
            routePoints = routePoints.filter(id => id !== node.id);
            activeShortestPath = null;
            activeRouteMeta = null;
        } else {
            if (routePoints.length >= 2) {
                routePoints.shift();
            }
            routePoints.push(node.id);
        }
        calculateResilience();
        if (routePoints.length === 2) {
            planRoute();
        } else {
            render();
        }
    }
}

function positionTooltip(canvasX, canvasY) {
    const tooltip = document.getElementById("telemetryTooltip");
    const rect = canvas.getBoundingClientRect();
    
    // Calculate layout position from 800x600 coords
    const scaleX = rect.width / 800;
    const scaleY = rect.height / 600;
    const domX = rect.left + (canvasX * scaleX);
    const domY = rect.top + (canvasY * scaleY);
    
    tooltip.style.left = (domX + 20) + "px";
    tooltip.style.top = (domY - 20) + "px";
    tooltip.style.display = "block";
}

function showNodeDetail(node) {
    const connections = activeGraph.edges.filter(e => e.from === node.id || e.to === node.id);
    const statusText = blockedNodes.has(node.id) ? 
        '<span class="badge badge-blocked">Blocked</span>' : 
        '<span class="badge badge-normal">Active</span>';
        
    const tooltip = document.getElementById("telemetryTooltip");
    const tooltipTitle = document.getElementById("tooltipTitle");
    const tooltipContent = document.getElementById("tooltipContent");
    
    // Restore standard monochromatic card style
    if (tooltip) {
        tooltip.style.background = "#070707";
        tooltip.style.border = "var(--border-thin)";
        tooltip.style.padding = "12px";
        tooltip.style.borderRadius = "4px";
        tooltip.style.boxShadow = "0 4px 20px rgba(0, 0, 0, 0.7)";
        tooltip.style.color = "";
        tooltip.style.textShadow = "";
    }
    
    tooltipTitle.textContent = `JUNCTION #${node.id}`;
    tooltipContent.innerHTML = `
        <div class="inspect-kv">
            <span class="lbl">Label</span>
            <span class="val">${node.name}</span>
        </div>
        <div class="inspect-kv">
            <span class="lbl">Status</span>
            <span class="val">${statusText}</span>
        </div>
        <div class="inspect-kv">
            <span class="lbl">Valency</span>
            <span class="val">${connections.length} links</span>
        </div>
        <div class="inspect-kv">
            <span class="lbl">Location</span>
            <span class="val">${node.x.toFixed(0)}m, ${node.y.toFixed(0)}m</span>
        </div>
        <div class="inspect-kv" style="border-top: 1px solid #222222; margin-top: 8px; padding-top: 8px;">
            <span class="lbl">Resilience Index</span>
            <span class="val">${activeGraph.stats.resilience || "100%"}</span>
        </div>
        <div class="inspect-kv">
            <span class="lbl">APLS Score</span>
            <span class="val">${activeGraph.stats.apls || "0.000"}</span>
        </div>
    `;
    
    positionTooltip(node.x, node.y);
}

function showRouteDetail() {
    const tooltipContent = document.getElementById("tooltipContent");
    const tooltipTitle = document.getElementById("tooltipTitle");
    
    tooltipTitle.textContent = "ROUTING DETAILS";
    
    if (routePoints.length === 0) {
        document.getElementById("telemetryTooltip").style.display = "none";
    } else if (routePoints.length === 1) {
        const startNode = activeGraph.nodes.find(n => n.id === routePoints[0]);
        if (startNode) {
            tooltipContent.innerHTML = `
                <div class="inspect-kv">
                    <span class="lbl">Start</span>
                    <span class="val">${startNode.name}</span>
                </div>
                <div style="font-size: 8px; color: var(--text-secondary); text-align: center; margin-top: 6px; font-weight: bold;">
                    SELECT END POINT ON MAP
                </div>
            `;
            positionTooltip(startNode.x, startNode.y);
        }
    } else if (routePoints.length === 2) {
        const startNode = activeGraph.nodes.find(n => n.id === routePoints[0]);
        const endNode = activeGraph.nodes.find(n => n.id === routePoints[1]);
        if (!startNode || !endNode) return;
        
        let pathStats = "";
        if (activeShortestPath) {
            let totalDist = 0;
            for (let i = 0; i < activeShortestPath.length - 1; i++) {
                totalDist += getDistanceBetweenNodes(activeShortestPath[i], activeShortestPath[i+1]);
            }
            const distanceKm = (totalDist * 0.005).toFixed(2);
            pathStats = `
                <div class="inspect-kv">
                    <span class="lbl">Route Status</span>
                    <span class="val" style="color: #ffffff; font-weight: bold;">SECURED</span>
                </div>
                <div class="inspect-kv">
                    <span class="lbl">Hops</span>
                    <span class="val">${activeShortestPath.length - 1} links</span>
                </div>
                <div class="inspect-kv">
                    <span class="lbl">Distance</span>
                    <span class="val">${distanceKm} km</span>
                </div>
            `;
        } else {
            pathStats = `
                <div class="inspect-kv">
                    <span class="lbl">Route Status</span>
                    <span class="val" style="color: #ef4444; font-weight: bold;">PARTITIONED</span>
                </div>
            `;
        }
        
        tooltipContent.innerHTML = `
            <div class="inspect-kv">
                <span class="lbl">Start</span>
                <span class="val">${startNode.name}</span>
            </div>
            <div class="inspect-kv">
                <span class="lbl">End</span>
                <span class="val">${endNode.name}</span>
            </div>
            ${pathStats}
            <div class="inspect-kv" style="border-top: 1px solid #222222; margin-top: 8px; padding-top: 8px;">
                <span class="lbl">Resilience Index</span>
                <span class="val">${activeGraph.stats.resilience || "100%"}</span>
            </div>
            <div class="inspect-kv">
                <span class="lbl">APLS Score</span>
                <span class="val">${activeGraph.stats.apls || "0.000"}</span>
            </div>
        `;
        positionTooltip(endNode.x, endNode.y);
    }
}

function showPlaceholderDetail() {
    const tooltip = document.getElementById("telemetryTooltip");
    if (tooltip) tooltip.style.display = "none";
}
