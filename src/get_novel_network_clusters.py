import pandas as pd
import ast 
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import glob
import networkx as nx
import community as community_louvain 
from collections import defaultdict
import matplotlib.cm as cm
import os 
import math


def filter_df(input_df, ref_list):
    filtered = input_df[(input_df['ref_tmscore']>= 0.5) & (input_df['target_tmscore']>= 0.5)]
    filtered_paralogs = list(set(filtered['reference'].unique()).union(set(filtered['target'].unique())))
    novel_list = list(set(filtered_paralogs) - set(ref_list))
    return filtered, novel_list 
    

# function to create the graph 
# returns graph G, and partition 
def create_louvain_graph(input_df):
    G = nx.Graph()

    # Add all unique nodes from the both ref and target 
    all_nodes = pd.unique(input_df[['reference', 'target']].values.ravel())
    print(len(all_nodes))
    G.add_nodes_from(all_nodes)

    for _, row in input_df.iterrows():
        ref = row['reference']
        tgt = row['target']
        # ref_len = row['ref_len']
        # tgt_len = row['target_len']

       
        if (ref != tgt and row['ref_tmscore'] >= 0.5 and row['target_tmscore']>= 0.5):
            G.add_edge(row['reference'], row['target'])
        # # # use this if only tm-score is used 
        # if (ref != tgt and row['ref_tmscore'] >= 0.5):
        #     G.add_edge(row['reference'], row['target'])
            
    # Apply Louvain clustering 
    partition = community_louvain.best_partition(G, random_state=42)

    # Ensure all nodes get partition
    for node in G.nodes:
        if node not in partition:
            partition[node] = -1 # assigns a default or outlier cluster 
            
    # remove the edges if found weak connections 
    # here remove if number of edges is less than 3
    node_degrees = dict(G.degree())
    outliers = {node for node, deg in node_degrees.items() if deg == 0}

    for node in outliers:
        partition[node] = -1

    # remove outlier nodes from the cluster 
    for node in outliers:
        for neighbor in list(G.neighbors(node)):
            G.remove_edge(node, neighbor)
    G.remove_nodes_from(outliers)

    # remove the nodes and clusters if size of cluster is less than 5
    # traverse through the parition of the graph 
    clusters = defaultdict(list)
    for node, cluseter_id in partition.items():
        clusters[cluseter_id].append(node)

    small_clusters = {cid for cid, nodes in clusters.items() if len(nodes)<8}
    ## this code if want to remove nodes here from the small cluster 
    # nodes_to_remove = []
    # for cid in small_clusters:
    #     nodes_to_remove.extend(clusters[cid])
    # G.remove_nodes_from(nodes_to_remove)

    # remove these nodes from partition as well
    large_partition = {
        node: cluseter_id
        for node, cluseter_id in partition.items()
        if node in G.nodes and cluseter_id not in small_clusters and cluseter_id != -1
    }
    small_partition = {
        node: cluseter_id
        for node, cluseter_id in partition.items()
        if node in G.nodes and cluseter_id in small_clusters and cluseter_id != -1
    }

    print(f"Remaining size of G is {len(G.nodes)}")
    num_outliers = sum(1 for v in partition.values() if v == -1)
    print(f"Number of nodes reassigned as outliers (degree == 0): {num_outliers}")
    return G, large_partition, small_partition


# function to mix with white
import matplotlib.colors as mcolors 
import matplotlib.cm as cm
import colorsys
    
# function to plot the graph 
def plot_graph(G, partition, novel_list, save_dir, filename):
    os.makedirs(save_dir, exist_ok=True)

    # keep only nodes present in the partition
    nodes_to_remove = [n for n in G.nodes if n not in partition]
    if nodes_to_remove:
        G = G.copy()
        G.remove_nodes_from(nodes_to_remove)

    novel_set = set(novel_list) & set(G.nodes)

    # group nodes by cluster (using the provided partition)
    cluster_to_members = defaultdict(list)
    for n in G.nodes:
        cluster_to_members[partition[n]].append(n)

    # keep only clusters that contain at least one novel node
    kept_clusters = [cid for cid, members in cluster_to_members.items()
                     if any(n in novel_set for n in members)]
    if not kept_clusters:
        print("No clusters contain nodes from novel_list; nothing to plot.")
        return

    kept_clusters = sorted(kept_clusters)
    kept_nodes = [n for n in G.nodes if partition[n] in kept_clusters]
    Gp = G.subgraph(kept_nodes).copy()
    part_kept = {n: partition[n] for n in Gp.nodes}

    # pick one non-novel label per kept cluster (degree-based representative)
    labels_to_draw = {}
    for cid in kept_clusters:
        members = [n for n in cluster_to_members[cid] if n in Gp]
        non_novel = [n for n in members if n not in novel_set]
        if non_novel:
            rep = max(non_novel, key=lambda x: Gp.degree(x))
            labels_to_draw[rep] = rep

    # ---- colors (mellow toward white) ----
    def mix_with_white(rgb, mix=0.5):
        r, g, b = mcolors.to_rgb(rgb)
        return (r + (1 - r) * mix, g + (1 - g) * mix, b + (1 - b) * mix)

    n_clusters = len(kept_clusters)
    if n_clusters <= 12:
        base = [cm.Paired(i % cm.Paired.N)[:3] for i in range(n_clusters)]
    else:
        base = [cm.tab20(i % cm.tab20.N)[:3] for i in range(n_clusters)]
    mellow = [mcolors.to_hex(mix_with_white(c, mix=0.5)) for c in base]
    color_map = {cid: mellow[i] for i, cid in enumerate(kept_clusters)}

    # ---- non-overlapping layout: local layout per cluster + grid packing ----
    # 1) local layout per cluster (normalize around 0 and scale by radius ~ sqrt(size))
    local_pos = {}            # per-node local (cluster-centered) coords
    cluster_radius = {}       # per-cluster radius before global grid placement
    for cid in kept_clusters:
        members = [n for n in Gp.nodes if part_kept[n] == cid]
        Hc = Gp.subgraph(members)

        # local layout; kamada-kawai is stable for small subgraphs
        lp = nx.kamada_kawai_layout(Hc)

        # center and normalize to unit radius
        coords = np.array([lp[n] for n in members])
        center = coords.mean(axis=0)
        coords -= center
        max_norm = np.max(np.linalg.norm(coords, axis=1)) if len(coords) else 1.0
        if max_norm == 0:
            max_norm = 1.0

        # choose radius ~ sqrt(size) (add 1 so singletons aren't tiny)
        r = math.sqrt(len(members) + 1.0)
        # save
        cluster_radius[cid] = r
        # scale to radius r
        coords = coords * (r / max_norm)
        for i, n in enumerate(members):
            local_pos[n] = coords[i]

    # 2) grid packing: place each cluster's center on a grid cell with spacing
    max_r = max(cluster_radius.values()) if cluster_radius else 1.0
    margin = 0.6 * max_r            # extra padding between clusters
    cell = 2 * max_r + margin       # cell size guarantees no overlap
    cols = math.ceil(math.sqrt(n_clusters))
    rows = math.ceil(n_clusters / cols)

    # center grid around (0,0) for a balanced view
    x0 = - (cols - 1) * cell / 2.0
    y0 = - (rows - 1) * cell / 2.0

    cluster_centers = {}
    for idx, cid in enumerate(kept_clusters):
        r = idx // cols
        c = idx % cols
        cluster_centers[cid] = np.array([x0 + c * cell, y0 + (rows - 1 - r) * cell])

    # 3) compose final global positions (local position + cluster center), then scale
    pos = {}
    global_scale = 3500.0  # adjust for figure size; bigger = more spread out
    for n in Gp.nodes:
        cid = part_kept[n]
        pos[n] = (local_pos[n] + cluster_centers[cid]) * global_scale

    # ---- draw ----
    plt.figure(figsize=(10, 8))
    ax = plt.gca()

    # edges first
    nx.draw_networkx_edges(Gp, pos, alpha=0.6, width=0.4, edge_color='gray', ax=ax)

    # non-novel nodes
    non_novel_nodes = [n for n in Gp.nodes if n not in novel_set]
    nx.draw_networkx_nodes(
        Gp, pos,
        nodelist=non_novel_nodes,
        node_size=[max(80, Gp.degree(n) * 20) for n in non_novel_nodes],
        node_color=[color_map[part_kept[n]] for n in non_novel_nodes],
        edgecolors='black',
        linewidths=0.3,
        alpha=0.9,
        ax=ax
    )

    # novel nodes (highlighted)
    highlighted = [n for n in Gp.nodes if n in novel_set]
    nx.draw_networkx_nodes(
        Gp, pos,
        nodelist=highlighted,
        node_size=300,
        node_color='#ffff33',
        edgecolors=[color_map[part_kept[n]] for n in highlighted],
        linewidths=2.5,
        alpha=0.95,
        ax=ax
    )

    # one label per cluster
    if labels_to_draw:
        nx.draw_networkx_labels(
            Gp, pos, labels=labels_to_draw,
            font_size=11, font_weight='normal', ax=ax
        )

    plt.axis('off')
    plt.tight_layout(pad=1.0)
    outpath = os.path.join(save_dir, f"{filename}.svg")
    plt.savefig(outpath, dpi=900, bbox_inches='tight')
    plt.show()
    print(f"Saved: {outpath}")