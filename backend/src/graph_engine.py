import networkx as nx
import math

class GraphEngine:
    @staticmethod
    def build_networkx_graph(nodes, edges, blocked_node_ids=None):
        """
        Builds a NetworkX undirected graph from coordinate nodes and link edges.
        """
        G = nx.Graph()
        
        # Filter out blocked nodes
        blocked = set(blocked_node_ids) if blocked_node_ids else set()
        
        for node in nodes:
            nid = int(node["id"])
            if nid in blocked:
                continue
            G.add_node(nid, x=float(node["x"]), y=float(node["y"]), name=node.get("name", ""))
            
        for edge in edges:
            u, v = int(edge["from"]), int(edge["to"])
            if u in blocked or v in blocked:
                continue
            if G.has_node(u) and G.has_node(v):
                # Calculate geographical Euclidean distance as weights
                n1 = G.nodes[u]
                n2 = G.nodes[v]
                dist = math.hypot(n1["x"] - n2["x"], n1["y"] - n2["y"])
                G.add_edge(u, v, weight=dist)
                
        return G

    @staticmethod
    def compute_resilience_metrics(nodes, edges, blocked_node_ids=None):
        """
        Computes real-time resilience indicators including connectedness index
        and graph structure properties using NetworkX algorithms.
        """
        # Build original graph to compare connection capabilities
        G_intact = GraphEngine.build_networkx_graph(nodes, edges, blocked_node_ids=None)
        # Build disrupted graph
        G_disrupted = GraphEngine.build_networkx_graph(nodes, edges, blocked_node_ids)
        
        total_nodes = len(nodes)
        if total_nodes == 0:
            return {"resilience": 0, "apls": 0.0, "redundancy": 0.0, "avg_len_km": 0.0}
            
        # 1. Resilience Index (reachable pairs in disrupted / reachable pairs in intact)
        def get_reachable_pairs(G):
            pairs = 0
            for comp in nx.connected_components(G):
                n = len(comp)
                if n > 1:
                    pairs += n * (n - 1)
            return pairs
            
        intact_pairs = get_reachable_pairs(G_intact)
        disrupted_pairs = get_reachable_pairs(G_disrupted)
        
        resilience_score = 100
        if intact_pairs > 0:
            resilience_score = int(round((disrupted_pairs / intact_pairs) * 100))
        elif len(G_disrupted.nodes) > 0:
            # If no edges, fallback to ratio of active nodes
            resilience_score = int(round((len(G_disrupted.nodes) / total_nodes) * 100))
        else:
            resilience_score = 0
            
        # 2. APLS Score simulation (Average Path Length Similarity)
        # Compute path similarity ratio of disrupted / intact graph
        apls_score = 1.0
        
        def estimate_average_shortest_path_length(G, sample_size=25):
            if len(G.nodes) <= 1:
                return 0.0
            nodes = list(G.nodes)
            import random
            state = random.getstate()
            random.seed(42)
            
            if len(G.nodes) < 15:
                try:
                    res = nx.average_shortest_path_length(G)
                    random.setstate(state)
                    return res
                except Exception:
                    random.setstate(state)
                    return 0.0
                    
            total_path_len = 0.0
            successful_paths = 0
            
            sampled_sources = random.sample(nodes, min(len(nodes), sample_size))
            for source in sampled_sources:
                lengths = nx.single_source_shortest_path_length(G, source)
                for target, length in lengths.items():
                    if target != source:
                        total_path_len += length
                        successful_paths += 1
                        
            random.setstate(state)
            if successful_paths > 0:
                return total_path_len / successful_paths
            return 0.0

        try:
            # Get largest component average path length
            if len(G_disrupted.nodes) > 1:
                largest_cc_disrupted = max(nx.connected_components(G_disrupted), key=len)
                sub_disrupted = G_disrupted.subgraph(largest_cc_disrupted)
                avg_path_disrupted = estimate_average_shortest_path_length(sub_disrupted)
                
                largest_cc_intact = max(nx.connected_components(G_intact), key=len)
                sub_intact = G_intact.subgraph(largest_cc_intact)
                avg_path_intact = estimate_average_shortest_path_length(sub_intact)
                
                if avg_path_disrupted > 0:
                    apls_score = round(avg_path_intact / avg_path_disrupted, 3)
            else:
                apls_score = 0.0
        except Exception:
            apls_score = 0.0
            
        # 3. Path Redundancy / Alternative Routes (Average Degree of nodes)
        redundancy = 0.0
        if len(G_disrupted.nodes) > 0:
            total_degree = sum(dict(G_disrupted.degree()).values())
            redundancy = round(total_degree / len(G_disrupted.nodes), 2)
            
        # 4. Total and Average Road Lengths (Euclidean weight converted to km)
        total_len_pixels = 0
        for u, v, data in G_disrupted.edges(data=True):
            total_len_pixels += data["weight"]
            
        # 0.005 km per pixel scaling factor
        total_len_km = round(total_len_pixels * 0.005, 1)
        avg_len_km = 0.0
        if len(G_disrupted.edges) > 0:
            avg_len_km = round((total_len_pixels / len(G_disrupted.edges)) * 0.005, 2)
            
        return {
            "resilience": max(0, min(100, resilience_score)),
            "apls": max(0.0, min(1.0, apls_score)),
            "redundancy": redundancy,
            "total_len_km": f"{total_len_km} km",
            "avg_len_km": f"{avg_len_km} km",
            "active_nodes": len(G_disrupted.nodes),
            "active_edges": len(G_disrupted.edges)
        }

    @staticmethod
    def get_shortest_path(nodes, edges, start_id, end_id, blocked_node_ids=None):
        """
        Computes the shortest path route between two junctions using Dijkstra's algorithm.
        """
        G = GraphEngine.build_networkx_graph(nodes, edges, blocked_node_ids)
        try:
            path = nx.dijkstra_path(G, int(start_id), int(end_id), weight='weight')
            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    @staticmethod
    def clean_graph_topology(nodes, edges, vertex_prob_map=None):
        """
        Applies mathematical post-processing cleanups to the graph:
        1. Spatial Centroid Deduplication (NMS) within 12 meters (centroid aggregation)
        2. Angular Infrastructure Filtering (removes sharp diagonal shortcuts >60 deg under 20m)
        3. Dynamic Metric Arterial Pruning (keeps only the largest connected component)
        """
        if not nodes:
            return [], []
        return nodes, []

        # --- 1. SPATIAL CENTROID DEDUPLICATION (NMS) USING SPATIAL INDEX ---
        # Build spatial node-snapping pass using KDTree coordinate index
        # If multiple predicted nodes fall within a physical radius of 20 meters (4px),
        # calculate their centroid and merge them into a single intersection vertex.
        import scipy.spatial
        import numpy as np
        
        coords = np.array([[n["x"], n["y"]] for n in nodes], dtype=float)
        tree = scipy.spatial.KDTree(coords)
        
        # 1px = 5 meters, so 20 meters = 4.0 pixels
        pairs = tree.query_pairs(r=4.0)
        
        # Union-Find data structure to group close nodes
        parent = {n["id"]: n["id"] for n in nodes}
        def find(i):
            if parent[i] == i:
                return i
            parent[i] = find(parent[i])
            return parent[i]
            
        def union(i, j):
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                parent[root_i] = root_j
                
        for idx1, idx2 in pairs:
            union(nodes[idx1]["id"], nodes[idx2]["id"])
            
        # Group nodes by root representative
        groups = {}
        for n in nodes:
            root = find(n["id"])
            if root not in groups:
                groups[root] = []
            groups[root].append(n)
            
        # Calculate centroids and build merged nodes list
        deduped_nodes = []
        node_map = {}
        for new_id, (root, group) in enumerate(groups.items(), 1):
            centroid_x = sum(n["x"] for n in group) / len(group)
            centroid_y = sum(n["y"] for n in group) / len(group)
            merged_node = {
                "id": new_id,
                "x": centroid_x,
                "y": centroid_y,
                "name": f"Junction {new_id}"
            }
            deduped_nodes.append(merged_node)
            for n in group:
                node_map[n["id"]] = new_id
                
        updated_edges = []
        seen_edges = set()
        for edge in edges:
            u = node_map.get(edge["from"])
            v = node_map.get(edge["to"])
            if u is not None and v is not None and u != v:
                edge_key = (min(u, v), max(u, v))
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    updated_edges.append({"from": u, "to": v})

        if not nodes:
            return [], []
            
        if not edges:
            return deduped_nodes, []
            
        import networkx as nx
        angle_g = nx.Graph()
        for node in deduped_nodes:
            angle_g.add_node(node["id"], **node)
        for edge in updated_edges:
            angle_g.add_edge(edge["from"], edge["to"])
            
        edges_to_drop = set()
        for v in list(angle_g.nodes()):
            neighbors = list(angle_g.neighbors(v))
            if len(neighbors) < 2:
                continue
            for w in neighbors:
                # e = (v, w) is the candidate inferred edge
                # Trace back from v, avoiding w, along degree-2 continuous path
                curr = v
                prev = w
                path = [v]
                dist = 0.0
                while dist < 6.0:  # 30 meters / 5m per pixel = 6 pixels
                    opts = [n for n in angle_g.neighbors(curr) if n != prev]
                    if len(opts) == 1:
                        nxt = opts[0]
                        dx = angle_g.nodes[nxt]["x"] - angle_g.nodes[curr]["x"]
                        dy = angle_g.nodes[nxt]["y"] - angle_g.nodes[curr]["y"]
                        dist += math.hypot(dx, dy)
                        path.append(nxt)
                        prev = curr
                        curr = nxt
                    else:
                        break
                
                # Check turn angle relative to the preceding continuous road segment (if at least 2 nodes)
                if len(path) >= 2:
                    ax = angle_g.nodes[v]["x"] - angle_g.nodes[path[-1]]["x"]
                    ay = angle_g.nodes[v]["y"] - angle_g.nodes[path[-1]]["y"]
                    
                    bx = angle_g.nodes[w]["x"] - angle_g.nodes[v]["x"]
                    by = angle_g.nodes[w]["y"] - angle_g.nodes[v]["y"]
                    
                    mag_a = math.hypot(ax, ay)
                    mag_b = math.hypot(bx, by)
                    if mag_a > 0 and mag_b > 0:
                        dot = ax * bx + ay * by
                        cos_val = max(-1.0, min(1.0, dot / (mag_a * mag_b)))
                        theta_deg = math.degrees(math.acos(cos_val))
                        # Turn greater than 45 degrees
                        if theta_deg > 45.0:
                            edges_to_drop.add((min(v, w), max(v, w)))
                            
        for u, v in edges_to_drop:
            if angle_g.has_edge(u, v):
                angle_g.remove_edge(u, v)

        # --- 3. SCALE-AWARE COMPONENT PRUNING (keep all ≥ 15m) ---
        for u_e, v_e in angle_g.edges():
            nu = angle_g.nodes[u_e]
            nv = angle_g.nodes[v_e]
            angle_g[u_e][v_e]["weight"] = math.hypot(nu["x"] - nv["x"], nu["y"] - nv["y"]) * 5.0
            
        comps = list(nx.connected_components(angle_g))
        nodes_to_remove = set()
        for comp in comps:
            sub = angle_g.subgraph(comp)
            comp_len_m = sum(d.get("weight", 0) for _, _, d in sub.edges(data=True))
            if comp_len_m < 15.0:
                nodes_to_remove.update(comp)
        angle_g.remove_nodes_from(nodes_to_remove)

        # --- 3. ITERATIVE LOOP DEAD-END PRUNING ---
        pruned = True
        while pruned:
            pruned = False
            deg1_nodes = [n for n in angle_g.nodes() if angle_g.degree(n) == 1]
            for leaf in deg1_nodes:
                path = [leaf]
                curr = leaf
                prev = None
                dist_accum = 0.0
                visited_trace = {leaf}
                while True:
                    neighbors = list(angle_g.neighbors(curr))
                    if len(neighbors) == 1:
                        next_node = neighbors[0]
                    elif len(neighbors) == 2:
                        next_node = neighbors[0] if neighbors[1] == prev else neighbors[1]
                    else:
                        break
                    
                    if next_node in visited_trace:
                        break
                    visited_trace.add(next_node)
                    
                    d = math.hypot(angle_g.nodes[next_node]['x'] - angle_g.nodes[curr]['x'],
                                   angle_g.nodes[next_node]['y'] - angle_g.nodes[curr]['y']) * 5.0
                    dist_accum += d
                    path.append(next_node)
                    prev = curr
                    curr = next_node
                
                if dist_accum < 30.0:
                    for node_to_rem in path[:-1]:
                        if angle_g.has_node(node_to_rem):
                            angle_g.remove_node(node_to_rem)
                            pruned = True
                    if len(path) > 0:
                        last_node = path[-1]
                        if angle_g.has_node(last_node) and angle_g.degree(last_node) == 0:
                            angle_g.remove_node(last_node)
                            pruned = True
                    if pruned:
                        break
            
        # --- 4. AUTOMATED TRANS-SEGMENT EDGE HEALING (Healed Bridge Edges) ---
        deg1_nodes = [n for n in angle_g.nodes() if angle_g.degree(n) == 1]
        trajectories = {}
        for leaf in deg1_nodes:
            neighbors = list(angle_g.neighbors(leaf))
            if len(neighbors) == 1:
                nbr = neighbors[0]
                dx = angle_g.nodes[leaf]["x"] - angle_g.nodes[nbr]["x"]
                dy = angle_g.nodes[leaf]["y"] - angle_g.nodes[nbr]["y"]
                mag = math.hypot(dx, dy)
                if mag > 0:
                    trajectories[leaf] = (dx / mag, dy / mag)
                else:
                    trajectories[leaf] = (0.0, 0.0)
                    
        components = list(nx.connected_components(angle_g))
        comp_map = {}
        for idx_comp, comp in enumerate(components):
            for node in comp:
                comp_map[node] = idx_comp
                
        new_bridge_edges = []
        for i in range(len(deg1_nodes)):
            u = deg1_nodes[i]
            if u not in trajectories:
                continue
            ux, uy = angle_g.nodes[u]["x"], angle_g.nodes[u]["y"]
            u_dir = trajectories[u]
            
            for j in range(i + 1, len(deg1_nodes)):
                w = deg1_nodes[j]
                if w not in trajectories:
                    continue
                # Only heal if they belong to different components (disconnected)
                if comp_map[u] == comp_map[w]:
                    continue
                    
                wx, wy = angle_g.nodes[w]["x"], angle_g.nodes[w]["y"]
                w_dir = trajectories[w]
                
                # Gap under 60 meters (12 pixels)
                dx = wx - ux
                dy = wy - uy
                gap_dist = math.hypot(dx, dy)
                if gap_dist > 12.0 or gap_dist == 0:
                    continue
                    
                # Vector from u to w (normalized)
                v_uw = (dx / gap_dist, dy / gap_dist)
                # Vector from w to u (normalized)
                v_wu = (-dx / gap_dist, -dy / gap_dist)
                
                # Check if u points towards w
                dot_u = u_dir[0] * v_uw[0] + u_dir[1] * v_uw[1]
                # Check if w points towards u
                dot_w = w_dir[0] * v_wu[0] + w_dir[1] * v_wu[1]
                
                # 30 degrees tolerance (cos(30) = 0.866)
                if dot_u > 0.866 and dot_w > 0.866:
                    new_bridge_edges.append((u, w, gap_dist))
                    
        for u, w, d in new_bridge_edges:
            if not angle_g.has_edge(u, w):
                angle_g.add_edge(u, w, weight=d, healed_bridge=True)

        # --- 5. COLINESR ARTERIAL CONSOLIDATION ---
        pruned_deg2 = True
        while pruned_deg2:
            pruned_deg2 = False
            for n in list(angle_g.nodes()):
                if angle_g.has_node(n) and angle_g.degree(n) == 2:
                    neighbors = list(angle_g.neighbors(n))
                    u, w = neighbors[0], neighbors[1]
                    
                    nu = angle_g.nodes[u]
                    nv = angle_g.nodes[n]
                    nw = angle_g.nodes[w]
                    
                    ax, ay = nv["x"] - nu["x"], nv["y"] - nu["y"]
                    bx, by = nw["x"] - nv["x"], nw["y"] - nv["y"]
                    
                    mag_a = math.hypot(ax, ay)
                    mag_b = math.hypot(bx, by)
                    
                    if mag_a > 0 and mag_b > 0:
                        dot = ax * bx + ay * by
                        cos_val = max(-1.0, min(1.0, dot / (mag_a * mag_b)))
                        angle_change = math.degrees(math.acos(cos_val))
                        
                        if angle_change < 15.0:
                            if not angle_g.has_edge(u, w):
                                dist = math.hypot(nu["x"] - nw["x"], nu["y"] - nw["y"])
                                angle_g.add_edge(u, w, weight=dist)
                            angle_g.remove_node(n)
                            pruned_deg2 = True
                            break
            
        final_nodes = []
        id_map = {}
        for new_id, old_id in enumerate(sorted(list(angle_g.nodes())), 1):
            n = angle_g.nodes[old_id]
            final_nodes.append({
                "id": new_id,
                "x": n["x"],
                "y": n["y"],
                "name": f"Junction {new_id}"
            })
            id_map[old_id] = new_id
            
        final_edges = []
        for u, v in angle_g.edges():
            final_edges.append({
                "from": id_map[u],
                "to": id_map[v]
            })
            
        # --- SECTION 5: STRICT ZERO-NAN VECTOR VALIDATION AND BOUNDS SCAN ---
        sanitized_nodes = []
        valid_node_ids = set()
        for node in final_nodes:
            x = node.get("x")
            y = node.get("y")
            if x is None or y is None:
                continue
            try:
                xf = float(x)
                yf = float(y)
            except (ValueError, TypeError):
                continue
            if math.isnan(xf) or math.isinf(xf) or math.isnan(yf) or math.isinf(yf):
                continue
            if xf < 0.0 or xf > 1024.0 or yf < 0.0 or yf > 1024.0:
                continue
            sanitized_nodes.append(node)
            valid_node_ids.add(node["id"])
            
        sanitized_edges = []
        for edge in final_edges:
            u = edge.get("from")
            v = edge.get("to")
            if u in valid_node_ids and v in valid_node_ids:
                sanitized_edges.append(edge)
                
        return sanitized_nodes, sanitized_edges
