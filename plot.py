import torch
from matplotlib import pyplot as plt
import datetime
import networkx as nx
import numpy as np
from torch_geometric.utils import k_hop_subgraph
import os
from dataset_preprocess import CoraDataset, PlanetoidDataset
import pickle
import utils

def viz_k_hop_op(edge_index, labels, target_node, hop, fig_path, fig_name):
    G = nx.Graph()
    G.add_nodes_from(range(labels.shape[0]))
    G.add_edges_from(edge_index.numpy().T)
    edge_list = torch.tensor(list(G.edges)).t()
    sub_nodes = k_hop_subgraph(target_node, hop, torch.cat([edge_list, edge_list.flip(0)], dim=1))[0]

    sub_G = G.subgraph(sub_nodes.numpy())

    print("size of sub graph node ",len(sub_nodes), " size of subgrpah edge ", sub_G.number_of_edges())

    plt.figure(figsize=(3.5, 3.5), dpi=80)
    pos = nx.spring_layout(sub_G, seed=7)
    nx.draw(sub_G,
            pos=pos,
            # with_labels=True,
            font_size=4,
            vmin=0,
            vmax=8,
            node_color=labels[sub_nodes].tolist(),
            edge_vmin=0.0,
            edge_vmax=1.0,
            width=0.5,
            node_size=450,
            alpha=0.9, )
    G_labels = sub_nodes.tolist()
    # G_labels = {k: f'{k}: {sub_G.degree[k]}' for k in G_labels}
    G_labels = {k: f'{k}' for k in G_labels}
    nx.draw_networkx_labels(sub_G,
                            pos=pos,
                            labels=G_labels,
                            font_size=8,
                            font_color="whitesmoke"
                            # font_color="red"
                            )
    plt.savefig(f'{fig_path}/{fig_name}.png')
    plt.clf()