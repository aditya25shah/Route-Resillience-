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
        try:
            # Get largest component average path length
            if len(G_disrupted.nodes) > 1:
                largest_cc_disrupted = max(nx.connected_components(G_disrupted), key=len)
                sub_disrupted = G_disrupted.subgraph(largest_cc_disrupted)
                avg_path_disrupted = nx.average_shortest_path_length(sub_disrupted)
                
                largest_cc_intact = max(nx.connected_components(G_intact), key=len)
                sub_intact = G_intact.subgraph(largest_cc_intact)
                avg_path_intact = nx.average_shortest_path_length(sub_intact)
                
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

        # --- 1. SPATIAL CENTROID DEDUPLICATION (NMS) ---
        merge_groups = []
        visited = set()
        
        for i, n1 in enumerate(nodes):
            nid = n1["id"]
            if nid in visited:
                continue
            group = [n1]
            visited.add(nid)
            for j, n2 in enumerate(nodes):
                n2id = n2["id"]
                if n2id in visited:
                    continue
                # Physical distance check (1px = 5m, so 12m limit)
                dist_m = math.hypot(n1["x"] - n2["x"], n1["y"] - n2["y"]) * 5.0
                if dist_m < 12.0:
                    group.append(n2)
                    visited.add(n2id)
            merge_groups.append(group)
            
        deduped_nodes = []
        node_map = {}
        for new_id, group in enumerate(merge_groups, 1):
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

        # --- 2. ANGULAR INFRASTRUCTURE FILTERING ---
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
            for idx1 in range(len(neighbors)):
                for idx2 in range(idx1 + 1, len(neighbors)):
                    u = neighbors[idx1]
                    w = neighbors[idx2]
                    
                    nu = angle_g.nodes[u]
                    nv = angle_g.nodes[v]
                    nw = angle_g.nodes[w]
                    
                    L_uv = math.hypot(nv["x"] - nu["x"], nv["y"] - nu["y"]) * 5.0
                    L_vw = math.hypot(nw["x"] - nv["x"], nw["y"] - nv["y"]) * 5.0
                    
                    if L_uv < 20.0 or L_vw < 20.0:
                        # Vector calculation: a = v - u, b = w - v
                        ax, ay = nv["x"] - nu["x"], nv["y"] - nu["y"]
                        bx, by = nw["x"] - nv["x"], nw["y"] - nv["y"]
                        
                        mag_a = math.hypot(ax, ay)
                        mag_b = math.hypot(bx, by)
                        
                        if mag_a > 0 and mag_b > 0:
                            dot = ax * bx + ay * by
                            cos_val = max(-1.0, min(1.0, dot / (mag_a * mag_b)))
                            theta_deg = math.degrees(math.acos(cos_val))
                            
                            if theta_deg > 60.0:
                                if L_uv < 20.0:
                                    edges_to_drop.add((min(u, v), max(u, v)))
                                if L_vw < 20.0:
                                    edges_to_drop.add((min(v, w), max(v, w)))
                                    
        for u, v in edges_to_drop:
            if angle_g.has_edge(u, v):
                angle_g.remove_edge(u, v)

        # --- 3. SCALE-AWARE COMPONENT PRUNING (keep all ≥ 15m) ---
        # Do NOT keep only the largest component — that destroys valid road fragments.
        # Instead, compute cumulative edge length per component and keep any ≥ 15 meters.
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
            
        # --- 4. COLLINEAR PATH CONSOLIDATION (GAP BRIDGING) ---
        # Find all degree-1 nodes
        deg1_nodes = [n for n in angle_g.nodes() if angle_g.degree(n) == 1]
        new_bridges = []
        for i in range(len(deg1_nodes)):
            n1_id = deg1_nodes[i]
            for j in range(i + 1, len(deg1_nodes)):
                n2_id = deg1_nodes[j]
                
                n1 = angle_g.nodes[n1_id]
                n2 = angle_g.nodes[n2_id]
                
                # Spatial distance between them in meters
                dist_m = math.hypot(n1["x"] - n2["x"], n1["y"] - n2["y"]) * 5.0
                if dist_m < 25.0: # Less than 25 meters gap
                    # Get neighbors
                    neigh1 = list(angle_g.neighbors(n1_id))[0]
                    neigh2 = list(angle_g.neighbors(n2_id))[0]
                    
                    n1_prev = angle_g.nodes[neigh1]
                    n2_prev = angle_g.nodes[neigh2]
                    
                    # Vector for segment 1: from prev to n1
                    v1_x, v1_y = n1["x"] - n1_prev["x"], n1["y"] - n1_prev["y"]
                    # Vector for segment 2: from prev to n2
                    v2_x, v2_y = n2["x"] - n2_prev["x"], n2["y"] - n2_prev["y"]
                    # Vector for gap: from n1 to n2
                    gap_x, gap_y = n2["x"] - n1["x"], n2["y"] - n1["y"]
                    
                    mag_v1 = math.hypot(v1_x, v1_y)
                    mag_v2 = math.hypot(v2_x, v2_y)
                    mag_gap = math.hypot(gap_x, gap_y)
                    
                    if mag_v1 > 0 and mag_v2 > 0 and mag_gap > 0:
                        # Angle between v1 and gap
                        dot1 = v1_x * gap_x + v1_y * gap_y
                        cos1 = max(-1.0, min(1.0, dot1 / (mag_v1 * mag_gap)))
                        angle1 = math.degrees(math.acos(cos1))
                        
                        # Angle between v2 and gap
                        dot2 = v2_x * gap_x + v2_y * gap_y
                        cos2 = max(-1.0, min(1.0, dot2 / (mag_v2 * mag_gap)))
                        angle2 = math.degrees(math.acos(cos2))
                        
                        # Directional angle deviation under 15 degrees
                        if angle1 < 15.0 and angle2 < 15.0:
                            new_bridges.append((n1_id, n2_id))
                            
        for u, v in new_bridges:
            if not angle_g.has_edge(u, v):
                n1 = angle_g.nodes[u]
                n2 = angle_g.nodes[v]
                dist = math.hypot(n1["x"] - n2["x"], n1["y"] - n2["y"])
                angle_g.add_edge(u, v, weight=dist)

        # --- 5. HORIZONTAL/VERTICAL GRID DE-NOISING (INTERMEDIATE NODE PRUNING) ---
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
            
        return final_nodes, final_edges
