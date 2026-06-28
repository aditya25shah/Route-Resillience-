// Presets configuration metadata
const PRESETS = {
    sf: { imgUrl: "preset_sf.png" },
    boston: { imgUrl: "preset_boston.png" },
    newyork: { imgUrl: "preset_newyork.png" }
};

// Global App State
let activePreset = "sf";
let blockedNodes = new Set();
let currentMode = "inspect"; // inspect, block, route
let routePoints = []; // Source and Target nodes for Route Planner
let activeShortestPath = null; // Calculated by backend
let hoverNode = null;
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

// Initialize presets imagery
const presetImages = {};
function preloadPresetImages() {
    Object.keys(PRESETS).forEach(key => {
        presetImages[key] = new Image();
        presetImages[key].src = PRESETS[key].imgUrl;
        presetImages[key].onload = () => {
            if (activePreset === key && !customImageLoaded) {
                render();
            }
        };
    });
}

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

// Setup preset buttons
document.querySelectorAll(".preset-btn").forEach(btn => {
    btn.addEventListener("click", (e) => {
        const btnElem = e.currentTarget;
        document.querySelectorAll(".preset-btn").forEach(b => b.classList.remove("active"));
        btnElem.classList.add("active");
        loadPreset(btnElem.getAttribute("data-preset"));
    });
});

// Canvas Interaction Mouse Event Listeners
canvas.addEventListener("mousemove", onMouseMove);
canvas.addEventListener("click", onMouseClick);
canvas.addEventListener("mouseleave", () => {
    hoverNode = null;
    render();
});

// Initialize App
preloadPresetImages();
setTimeout(() => {
    loadPreset("sf");
}, 200);

// Function definitions
function resizeCanvas() {
    const width = wrapper.clientWidth;
    const height = wrapper.clientHeight;
    
    canvas.width = width;
    canvas.height = height;
    
    render();
}

function loadPreset(key) {
    customImageLoaded = false;
    activePreset = key;
    blockedNodes.clear();
    routePoints = [];
    activeShortestPath = null;
    
    if (systemStatus) systemStatus.textContent = "Loading preset...";
    
    // Draw loading overlay locally first
    if (loadingOverlay) loadingOverlay.classList.add("active");
    if (progressBarFill) progressBarFill.style.width = "20%";
    if (loadingStep) loadingStep.textContent = "Loading preset city raster tiles...";
    
    const img = presetImages[key];
    if (img && img.complete) {
        processPresetInference(img);
    } else {
        const fallbackImg = new Image();
        fallbackImg.src = PRESETS[key].imgUrl;
        fallbackImg.onload = () => {
            presetImages[key] = fallbackImg;
            processPresetInference(fallbackImg);
        };
    }
    
    showPlaceholderDetail();
}

function processPresetInference(img) {
    if (progressBarFill) progressBarFill.style.width = "40%";
    if (loadingStep) loadingStep.textContent = "Encoding preset image base64...";
    
    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = img.width || 800;
    tempCanvas.height = img.height || 600;
    const tempCtx = tempCanvas.getContext("2d");
    tempCtx.drawImage(img, 0, 0);
    const dataUrl = tempCanvas.toDataURL("image/png");
    
    triggerInferencePipeline(dataUrl);
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
    if (loadingOverlay) loadingOverlay.classList.add("active");
    if (progressBarFill) progressBarFill.style.width = "60%";
    if (loadingStep) loadingStep.textContent = "Running Sat2Graph GTE inference on backend...";
    if (systemStatus) systemStatus.textContent = "Running live inference...";
    
    fetch("/api/infer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: imgDataSrc })
    })
    .then(response => {
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
        console.error(err);
        if (loadingOverlay) loadingOverlay.classList.remove("active");
        if (systemStatus) systemStatus.textContent = "Inference pipeline offline";
        alert("Geospatial backend connection error. Please verify route-backend is running.");
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
            }
            render();
        }
    })
    .catch(err => console.error("Error fetching resilience metrics:", err));
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

// Render loop
function render() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    const showSat = chkSatellite ? chkSatellite.checked : true;
    const showMask = chkMask ? chkMask.checked : true;
    const showGraph = chkGraph ? chkGraph.checked : true;
    
    // 1. Draw Satellite Background
    if (showSat) {
        if (customImageLoaded && customImageSrc) {
            ctx.drawImage(customImageSrc, 0, 0, canvas.width, canvas.height);
        } else if (presetImages[activePreset]) {
            ctx.drawImage(presetImages[activePreset], 0, 0, canvas.width, canvas.height);
        }
    } else {
        // Dark blueprint background
        ctx.fillStyle = "#000000";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Grid pattern
        ctx.strokeStyle = "rgba(255, 255, 255, 0.02)";
        ctx.lineWidth = 1;
        const gridSz = 40;
        for (let x = 0; x < canvas.width; x += gridSz) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, canvas.height);
            ctx.stroke();
        }
        for (let y = 0; y < canvas.height; y += gridSz) {
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
            
            // Scaled endpoints based on window resize
            const p1 = getScaledCoords(n1);
            const p2 = getScaledCoords(n2);
            
            // Draw wide glowing pathway
            ctx.beginPath();
            ctx.moveTo(p1.x, p1.y);
            ctx.lineTo(p2.x, p2.y);
            ctx.strokeStyle = "rgba(255, 255, 255, 0.08)";
            ctx.lineWidth = 14;
            ctx.lineCap = "round";
            ctx.stroke();
            
            ctx.strokeStyle = "rgba(255, 255, 255, 0.15)";
            ctx.lineWidth = 8;
            ctx.stroke();
        });
        ctx.restore();
    }
    
    // 3. Draw Extracted Graph Overlay
    if (showGraph) {
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
            
            // Highlight connections if blocked
            if (blockedNodes.has(e.from) || blockedNodes.has(e.to)) {
                ctx.strokeStyle = "rgba(255, 255, 255, 0.15)";
                ctx.lineWidth = 2;
                ctx.setLineDash([4, 4]);
            } else {
                ctx.strokeStyle = "#ffffff";
                ctx.lineWidth = 2;
                ctx.setLineDash([]);
            }
            ctx.stroke();
        });
        ctx.setLineDash([]);
        
        // Draw shortest path
        if (currentMode === "route" && routePoints.length === 2 && activeShortestPath) {
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
                ctx.strokeStyle = "#ffffff";
                ctx.lineWidth = 5;
                ctx.shadowColor = "#ffffff";
                ctx.shadowBlur = 10;
                ctx.stroke();
            }
            ctx.restore();
        }
        
        // Draw Nodes
        activeGraph.nodes.forEach(node => {
            const p = getScaledCoords(node);
            
            ctx.beginPath();
            ctx.arc(p.x, p.y, 6, 0, 2 * Math.PI);
            
            let color = "#ffffff";
            let strokeColor = "rgba(255, 255, 255, 0.2)";
            let glow = false;
            
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
            
            ctx.fillStyle = color;
            if (glow) {
                ctx.save();
                ctx.shadowColor = color;
                ctx.shadowBlur = 8;
                ctx.fill();
                ctx.restore();
            } else {
                ctx.fill();
            }
            
            // Outer ring
            ctx.beginPath();
            ctx.arc(p.x, p.y, 10, 0, 2 * Math.PI);
            ctx.strokeStyle = strokeColor;
            ctx.lineWidth = 1.5;
            ctx.stroke();
        });
    }
}

// Convert absolute coordinates stored in graph (from original 800x600 coordinate grid) to current viewport coordinates
function getScaledCoords(node) {
    // Presets and generated custom graphs are designed relative to 800x600 aspect coordinates
    const scaleX = canvas.width / 800;
    const scaleY = canvas.height / 600;
    return {
        x: node.x * scaleX,
        y: node.y * scaleY
    };
}

// Reverse conversion (from screen client x,y to 800x600 absolute grid coords)
function getAbsoluteCoords(clientX, clientY) {
    const rect = canvas.getBoundingClientRect();
    const x = clientX - rect.left;
    const y = clientY - rect.top;
    
    const scaleX = 800 / rect.width;
    const scaleY = 600 / rect.height;
    return {
        x: x * scaleX,
        y: y * scaleY
    };
}

// Mouse Event handlers
function onMouseMove(e) {
    const coords = getAbsoluteCoords(e.clientX, e.clientY);
    
    // Find if hovering over a node
    let found = null;
    activeGraph.nodes.forEach(node => {
        const dist = Math.hypot(node.x - coords.x, node.y - coords.y);
        if (dist <= 16) { // slightly wider target hit zone
            found = node;
        }
    });
    
    if (found !== hoverNode) {
        hoverNode = found;
        render();
    }
}

function onMouseClick(e) {
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
        } else {
            if (routePoints.length >= 2) {
                routePoints.shift();
            }
            routePoints.push(node.id);
        }
        calculateResilience(); // This triggers backend updates and Dijkstra solution
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
        
    const tooltipTitle = document.getElementById("tooltipTitle");
    const tooltipContent = document.getElementById("tooltipContent");
    
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
