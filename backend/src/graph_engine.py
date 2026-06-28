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
