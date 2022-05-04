from torch import optim
import torch
from tqdm import tqdm
from torch.nn import functional as F
from sklearn import metrics

import scipy.sparse as sp
import numpy as np
from scipy.sparse.csgraph import connected_components
import pickle

from matplotlib import pyplot as plt
import datetime
import networkx as nx
from torch_geometric.utils import k_hop_subgraph
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def load_variable(filename):
    with open(filename, 'rb') as f:
        [var_list] = pickle.load(f)
    return var_list

def save_to_file(var_list, filename):
    with open(filename, 'wb') as f:
        pickle.dump(var_list, f)

def load_npz(file_name):
    """Load a SparseGraph from a Numpy binary file.

    Parameters
    ----------
    file_name : str
        Name of the file to load.

    Returns
    -------
    sparse_graph : gust.SparseGraph
        Graph in sparse matrix format.

    """
    if not file_name.endswith('.npz'):
        file_name += '.npz'
    with np.load(file_name, allow_pickle=True) as loader:
        if file_name.endswith("cora_ml.npz"):
            loader = dict(loader)['arr_0'].item()
        else:
            loader = dict(loader)
        adj_matrix = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                                              loader['adj_indptr']), shape=loader['adj_shape'])

        if 'attr_data' in loader:
            attr_matrix = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                                                   loader['attr_indptr']), shape=loader['attr_shape'])
        else:
            attr_matrix = None

        labels = loader.get('labels')

    return adj_matrix, attr_matrix, labels


def largest_connected_components(adj, n_components=1):
    """Select the largest connected components in the graph.

    Parameters
    ----------
    sparse_graph : gust.SparseGraph
        Input graph.
    n_components : int, default 1
        Number of largest connected components to keep.

    Returns
    -------
    sparse_graph : gust.SparseGraph
        Subgraph of the input graph where only the nodes in largest n_components are kept.

    """
    _, component_indices = connected_components(adj)
    component_sizes = np.bincount(component_indices)
    components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  # reverse order to sort descending
    nodes_to_keep = [
        idx for (idx, component) in enumerate(component_indices) if component in components_to_keep


    ]
    print("Selecting {0} largest connected components".format(n_components))
    return nodes_to_keep

def preprocess_graph(adj):
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = adj_.sum(1).A1
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5))
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).T.dot(degree_mat_inv_sqrt).tocsr()
    return adj_normalized

def train(model, data, model_path, lr=0.005, epochs=300, verbose=True):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    train_loss_values, train_acc_values = [], []
    test_loss_values, test_acc_values = [], []

    best = np.inf
    bad_counter = 0

    # data.x = data.x.to(device)
    data.edge_index = data.edge_index.to(device)

    for epoch in tqdm(range(epochs), desc='Training', leave=False):
        if epoch == 0:
            print()
            print('       |     Trainging     |     Validation     |')
            print('       |-------------------|--------------------|')
            print(' Epoch |  loss    accuracy |  loss    accuracy  |')
            print('-------|-------------------|--------------------|')

        train_loss, train_acc = train_on_epoch(model, optimizer, data)
        train_loss_values.append(train_loss.item())
        train_acc_values.append(train_acc.item())

        # test_loss, test_acc = evaluate(model, data, data.val_mask)
        test_acc, test_loss, _, _, _, _, _, _, = evaluate(model, data, data.val_mask)
        test_loss_values.append(test_loss.item())
        test_acc_values.append(test_acc)

        if test_loss_values[-1] < best:
            bad_counter = 0
            log = '  {:3d}  | {:.4f}    {:.4f}  | {:.4f}    {:.4f}   |'.format(epoch + 1,
                                                                               train_loss.item(),
                                                                               train_acc.item(),
                                                                               test_loss.item(),
                                                                               test_acc)

            torch.save({'state_dict': model.state_dict()}, model_path)
            log += ' save model to {}'.format(model_path)

            if verbose:
                tqdm.write(log)

            best = test_loss_values[-1]
        else:
            bad_counter += 1

    print('-------------------------------------------------')

    model.load_state_dict(torch.load(model_path)['state_dict'])

def train_on_epoch(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    output = model(data.x, data.edge_index, None)
    output = F.log_softmax(output, dim=1)
    train_loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
    train_acc = accuracy(output[data.train_mask], data.y[data.train_mask])

    train_loss.backward()
    optimizer.step()

    return train_loss, train_acc


def evaluate(model, data, mask):

    model.eval()
    with torch.no_grad():
        output = model(data.x, data.edge_index, None)
        loss = F.nll_loss(output[mask], data.y[mask])
        acc = accuracy(output[mask], data.y[mask]).item()

        _, pred = output[mask].max(dim=1)
        y, m, p = map(lambda x:x.detach().cpu(), [data.y, mask, pred])
        res = metrics.classification_report(y[m], p, output_dict=True)
        macro_precision = res['macro avg']['precision']
        macro_recall = res['macro avg']['recall']
        macro_f1_score = res['macro avg']['f1-score']
        weighted_precision = res['weighted avg']['precision']
        weighted_recall = res['weighted avg']['recall']
        weighted_f1_score = res['weighted avg']['f1-score']
        print(" acc is", acc, res['accuracy'])

    return acc, loss, macro_precision, macro_recall, macro_f1_score, weighted_precision, weighted_recall, weighted_f1_score

def accuracy(output, labels):
    _, pred = output.max(dim=1)
    correct = pred.eq(labels).double()
    correct = correct.sum()

    return correct / len(labels)



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
    plt.savefig(f'{fig_path}/{fig_name}_with_{str(len(sub_nodes))}_total_nodes.pdf')
    plt.clf()