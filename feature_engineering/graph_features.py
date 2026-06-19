import pandas as pd
import numpy as np
import heapq
from collections import deque

def build_transition_graph(df: pd.DataFrame) -> tuple:
    """
    Builds the historical transition graph between junctions.
    Nodes: Junctions
    Edges: Junction pairs connected by the same corridor
    Weight: Travel Cost = avg duration + closure rate * 100 + incident count / 10.0
    """
    df_clean = df.copy()
    df_clean['junction'] = df_clean['junction'].fillna('unknown')
    df_clean['corridor'] = df_clean['corridor'].fillna('unknown')
    
    # Calculate duration
    s_dt = pd.to_datetime(df_clean['start_datetime'], errors='coerce')
    e_dt = pd.to_datetime(df_clean['end_datetime'], errors='coerce')
    c_dt = pd.to_datetime(df_clean['closed_datetime'], errors='coerce')
    r_dt = pd.to_datetime(df_clean['resolved_datetime'], errors='coerce')
    end_dt = c_dt.fillna(r_dt).fillna(e_dt)
    df_clean['duration'] = (end_dt - s_dt).dt.total_seconds() / 60.0
    df_clean['duration'] = df_clean['duration'].fillna(30.0)
    df_clean['requires_road_closure'] = df_clean['requires_road_closure'].astype(bool)
    
    # Map coordinates
    junction_coords = df_clean.groupby('junction')[['latitude', 'longitude']].mean().to_dict('index')
    
    # Precompute corridor stats
    corridor_stats = df_clean.groupby('corridor').agg(
        avg_duration=('duration', 'mean'),
        closure_rate=('requires_road_closure', 'mean'),
        incidents=('id', 'count')
    ).to_dict('index')
    
    # Find all junctions per corridor and link them by coordinate order
    corridors = df_clean['corridor'].unique()
    adj = {}
    
    for corr in corridors:
        if corr == 'unknown':
            continue
        corr_juncs = df_clean[df_clean['corridor'] == corr]['junction'].unique()
        corr_juncs = [j for j in corr_juncs if j != 'unknown']
        if len(corr_juncs) < 2:
            continue
            
        # Sort junctions by average latitude to order them sequentially along the corridor
        corr_juncs_sorted = sorted(corr_juncs, key=lambda j: junction_coords.get(j, {'latitude': 0})['latitude'])
        
        # Connect sequential junctions
        stats = corridor_stats.get(corr, {'avg_duration': 30.0, 'closure_rate': 0.1, 'incidents': 1})
        cost = stats['avg_duration'] + stats['closure_rate'] * 100.0 + stats['incidents'] / 10.0
        
        for i in range(len(corr_juncs_sorted) - 1):
            u = corr_juncs_sorted[i]
            v = corr_juncs_sorted[i+1]
            
            if u not in adj: adj[u] = {}
            if v not in adj: adj[v] = {}
            
            # Keep minimum cost for parallel corridor links
            adj[u][v] = min(adj[u].get(v, float('inf')), cost)
            adj[v][u] = min(adj[v].get(u, float('inf')), cost)
            
    return adj, junction_coords

def compute_centralities(adj: dict) -> tuple:
    """
    Computes Degree, Closeness, and Betweenness Centralities using pure Python.
    """
    nodes = list(adj.keys())
    n = len(nodes)
    
    degree_centrality = {}
    closeness_centrality = {}
    betweenness_centrality = {node: 0.0 for node in nodes}
    
    if n <= 1:
        return {k: 0.0 for k in nodes}, {k: 0.0 for k in nodes}, {k: 0.0 for k in nodes}
        
    # 1. Degree Centrality
    for u in nodes:
        degree_centrality[u] = len(adj[u]) / (n - 1)
        
    # 2. Closeness Centrality & Betweenness Centrality (Brandes' Algorithm)
    for s in nodes:
        # Single-source shortest paths (BFS style path counting for betweenness)
        S = []
        P = {w: [] for w in nodes}
        sigma = {w: 0.0 for w in nodes}
        sigma[s] = 1.0
        d = {w: -1 for w in nodes}
        d[s] = 0
        q = deque([s])
        
        while q:
            v = q.popleft()
            S.append(v)
            for w in adj[v]:
                # Path discovery
                if d[w] < 0:
                    d[w] = d[v] + 1
                    q.append(w)
                if d[w] == d[v] + 1:
                    sigma[w] += sigma[v]
                    P[w].append(v)
                    
        # Closeness calculation for source s
        total_dist = sum(d[w] for w in nodes if d[w] > 0)
        reachable = sum(1 for w in nodes if d[w] > 0)
        if total_dist > 0:
            closeness_centrality[s] = (reachable / (n - 1)) * (reachable / total_dist)
        else:
            closeness_centrality[s] = 0.0
            
        # Betweenness accumulation
        delta = {w: 0.0 for w in nodes}
        while S:
            w = S.pop()
            for v in P[w]:
                delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != s:
                betweenness_centrality[w] += delta[w]
                
    # Normalize betweenness (divide by 2 for undirected)
    for u in nodes:
        betweenness_centrality[u] /= 2.0
        # Scale betweenness centrality to be in a reasonable range
        if n > 2:
            betweenness_centrality[u] /= ((n - 1) * (n - 2) / 2.0)
            
    return degree_centrality, closeness_centrality, betweenness_centrality

def run_dijkstra(adj: dict, start: str, target: str) -> tuple:
    """
    Computes shortest path from start to target junction.
    Returns (path: list, total_cost: float)
    """
    if start not in adj or target not in adj:
        return [], float('inf')
        
    queue = [(0.0, start, [start])]
    visited = set()
    
    while queue:
        cost, node, path = heapq.heappop(queue)
        
        if node in visited:
            continue
        visited.add(node)
        
        if node == target:
            return path, cost
            
        for neighbor, weight in adj[node].items():
            if neighbor not in visited:
                heapq.heappush(queue, (cost + weight, neighbor, path + [neighbor]))
                
    return [], float('inf')

_centralities_cache = {}

def add_graph_features(df: pd.DataFrame, adj: dict = None, train_mode: bool = True) -> tuple:
    """
    Maps precomputed graph centralities back into the feature matrix.
    """
    global _centralities_cache
    df = df.copy()
    junction_coords = {}
    
    if train_mode or adj is None:
        adj, junction_coords = build_transition_graph(df)
        
    adj_id = id(adj)
    if not train_mode and adj_id in _centralities_cache:
        deg, close, bet = _centralities_cache[adj_id]
    else:
        deg, close, bet = compute_centralities(adj)
        if not train_mode:
            _centralities_cache[adj_id] = (deg, close, bet)
    
    # Map centralities
    df['degree_centrality'] = df['junction'].map(deg).fillna(0.0)
    df['closeness_centrality'] = df['junction'].map(close).fillna(0.0)
    df['betweenness_centrality'] = df['junction'].map(bet).fillna(0.0)
    
    return df, adj, junction_coords
