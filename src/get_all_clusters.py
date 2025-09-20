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


def filter_data(input_df, ref_list):
    filtered = input_df[(input_df['ref_tmscore']>= 0.5) & (input_df['target_tmscore']>= 0.5)]
    filtered_paralogs = list(set(filtered['reference'].unique()).union(set(filtered['target'].unique())))
    novel_list = list(set(filtered_paralogs) - set(ref_list))
    return filtered, novel_list 
    

# function to create the graph 
# returns graph G, and partition 
def create_louvain_graph_all_cluster(input_df):
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
def plot_all_graph(G, partition, novel_list, save_dir, filename):
    nodes_to_remove = [node for node in G.nodes if node not in partition]
    G.remove_nodes_from(nodes_to_remove)
    filtered_novel = [p for p in novel_list if p in G.nodes]
    # to_show = [p for p in ids_to_show if p in G.nodes]
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import colorsys

    # Create cluster color map
    unique_clusters = sorted(set(partition.values()))
    n_clusters = len(unique_clusters)
    
    # group nodes by cluster (using the provided partition)
    cluster_to_members = defaultdict(list)
    for n in G.nodes:
        cluster_to_members[partition[n]].append(n)
    novel_set = set(novel_list)
        
    to_show = {}
    for cid, members in cluster_to_members.items():
        non_novel = [n for n in members if n not in novel_set]
        if non_novel:
            rep = max(non_novel, key=lambda x: G.degree(x))
            to_show[rep] = rep
            
    # function to mix with white
    import matplotlib.colors as mcolors 
    def mix_with_white(rgb, mix=0.5):
        r, g, b = mcolors.to_rgb(rgb)
        return (1- (1-r) * mix,
                1 - (1-g) * mix,
                1 - (1-b) * mix)

    def get_distinct_colors(n):
        colors = []
        for i in range(n):

            hue = i / float(n)
            saturation = 0.4 + 0.3 * (i % 3) # moderate saturation
            value = 0.9
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            if sum(rgb) < 2.0: # if too dark
                rgb = tuple(min(1, c * 1.3) for c in rgb)
            colors.append(mcolors.rgb2hex(rgb))
        return colors
    
    if n_clusters <= 12:
        base = [cm.Paired(i)[:3] for i in range(n_clusters)]
    else:
        base = [cm.tab20(i % cm.tab20.N)[:3] for i in range(n_clusters)]
        
    # make them mellow by mixing with white
    mellow = [mcolors.to_hex(mix_with_white(c, mix=0.5)) for c in base]
    color_map = {cluster_id: mellow[i] for i, cluster_id in enumerate(unique_clusters)}
    # Compute layout
    # pos = nx.spring_layout(G, seed=42, k=3, iterations=200, scale=1.5)
    pos = nx.kamada_kawai_layout(G)
    for n in pos:
        pos[n] = pos[n] * 15000

    # Cluster separation parameters
    RADIUS = 15000   # distance between cluster centers
    COMPRESSION = 0.7  # how much to compress each cluster (0 = tight, 1 = loose)

    angle_step = 4 * np.pi / n_clusters

    for i, cluster_id in enumerate(unique_clusters):
        members = [n for n in G.nodes if partition[n] == cluster_id]
        if len(members) > 1:
            # Determine center of layout
            cluster_center = np.mean([pos[n] for n in members], axis=0)
            
            # Position cluster around a circle
            angle = i * angle_step
            dx = RADIUS * np.cos(angle)
            dy = RADIUS * np.sin(angle)
            cluster_offset = np.array([dx, dy])
            
            # Compress around cluster center and apply offset
            for n in members:
                pos[n] = cluster_offset + cluster_center + COMPRESSION * (pos[n] - cluster_center)
    
    
    # Node styling 
    non_marked_nodes = [n for n in G.nodes if n not in novel_list]


    # Plot
    plt.figure(figsize=(10, 8))
    ax = plt.gca()

    #Draw_edges first 
    nx.draw_networkx_edges(
        G, pos,
        alpha=0.6,
        width=0.4,
        edge_color='gray',
        ax=ax
    )

    # Draw non-marked nodes
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=non_marked_nodes,
        node_size=[max(80, G.degree(n)*20) for n in non_marked_nodes],
        node_color=[color_map[partition[n]] for n in non_marked_nodes],
        # edgecolors=[color_map[partition[n]] for n in non_marked_nodes],
        edgecolors='black',
        linewidths=0.3,
        alpha=0.9,
        ax=ax
    )

    # Draw marked nodes on top with highlight 
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=filtered_novel,
        node_size=300,
        node_color='#ffff33',  # bright yellow
        edgecolors=[color_map[partition[n]] for n in filtered_novel],
        linewidths=2.5,
        alpha=0.95,
        ax=ax
    )

    # # Draw edges
    # nx.draw_networkx_edges(G, pos, alpha=0.2, width=0.2)

    # Optional: disable labels
    if to_show:
        labels = {node: node for node in to_show if node in G.nodes}
        nx.draw_networkx_labels(
            G, pos, labels=labels,
            font_size=11,
            font_weight='normal',
            ax=ax
        )
    
    plt.axis('off')
    plt.tight_layout(pad=1.0)
    outpath = os.path.join(save_dir, f"{filename}.svg")
    plt.savefig(outpath, dpi=900, bbox_inches='tight')
    plt.show()
    print(f"Saved: {outpath}")