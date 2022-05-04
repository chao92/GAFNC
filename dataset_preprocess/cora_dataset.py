import torch
import numpy as np
import os.path as osp
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data, InMemoryDataset, download_url
import utils


class CoraDataset(InMemoryDataset):
    r"""
    The Synthetic datasets used in
    `Parameterized Explainer for Graph Neural Network <https://arxiv.org/abs/2011.04573>`_.
    It takes Barabási–Albert(BA) graph or balance tree as base graph
    and randomly attachs specific motifs to the base graph.

    Args:
        root (:obj:`str`): Root data directory to save datasets
        name (:obj:`str`): The name of the dataset_preprocess. Including :obj:`BA_shapes`, BA_grid,
        transform (:obj:`Callable`, :obj:`None`): A function/transform that takes in an
            :class:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (:obj:`Callable`, :obj:`None`):  A function/transform that takes in
            an :class:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)

    """
    # Format: name: [display_name, url_name, filename]
    names = {
        'cora': ['cora', 'cora.npz', 'cora'],
        'cora-added': ['cora_added', 'cora_added.npz', 'cora_added'],
        'photo': ['photo', 'photo.npz', 'photo'],
        'photo-added': ['photo_added', 'photo_added.npz', 'photo_added']
    }

    """
    random_generate 0: feature is sample from origin graph
    random_generate 1: feature is generated from random
    """

    def __init__(self, root, name, random_generate=0, transform=None, pre_transform=None, added_new_nodes=None):
        self.name = name.lower()
        self.added_node_num = None
        self.train_index = None
        self.val_index = None
        self.test_index = None
        self.random_generate = random_generate
        if added_new_nodes:
            self.added_node_num = added_new_nodes
        else:
            self.added_node_num = 0
        super(CoraDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        if self.added_node_num:
            return osp.join(self.root, self.name, str(self.added_node_num), 'processed')
        else:
            return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return f'{self.names[self.name][2]}.npz'

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        url = self.url.format(self.names[self.name][1])
        print(" url is", url)
        path = download_url(url, self.raw_dir)

    def process(self):
        # Read data into huge `Data` list.
        data = self.preprocessing_cora_data()
        data = data if self.pre_transform is None else self.pre_transform(data)
        data_list = [data]

        torch.save(self.collate(data_list), self.processed_paths[0])

    def __repr__(self):
        return '{}({})'.format(self.names[self.name][0], len(self))

    def preprocessing_cora_data(self):
        """ Load a SparseGraph from a Numpy binary file.
        Returns:

        """
        print("self raw paths is", self.raw_paths[0])
        adj_matrix, attr_matrix, labels = utils.load_npz(self.raw_paths[0])
        # cora_ml shape:(2708, 2708) (2708, 1433) (2708,)
        # cora shape: (2485, 2485) (2485, 1433) (2485,)
        print("origin adj shape: ", adj_matrix.shape)

        adj_matrix = adj_matrix + adj_matrix.T
        adj_matrix[adj_matrix > 1] = 1

        lcc = utils.largest_connected_components(adj_matrix)

        adj_matrix = adj_matrix[lcc][:,lcc]
        assert np.abs(adj_matrix - adj_matrix.T).sum() == 0, "Input graph is not symmetric"
        assert adj_matrix.max() == 1 and len(np.unique(adj_matrix[adj_matrix.nonzero()].A1)) == 1, "Graph must be unweighted"
        assert adj_matrix.sum(0).A1.min() > 0, "Graph contains singleton nodes"

        attr_matrix = attr_matrix[lcc].astype('float32')
        labels = labels[lcc]

        attr_matrix = attr_matrix.toarray()
        adj_matrix = adj_matrix.toarray()
        feature_dim = attr_matrix.shape[1]
        if self.added_node_num:
            print(" ------ before add new nodes ------")
            print("adj shape: ", adj_matrix.shape, "att shape: ", attr_matrix.shape, "label shape: ", labels.shape)
            print(" adding total ", self.added_node_num, " nodes to the graph")


            for i in range(self.added_node_num):

                if self.random_generate == 0:
                    # method 1: sampled from original graph attr matrix
                    rd_index = np.random.randint(attr_matrix.shape[0],size = 1)
                    attr_matrix = np.append(attr_matrix, attr_matrix[rd_index], axis=0)
                    labels = np.append(labels, labels[rd_index], axis=0)
                elif self.random_generate == 1:
                    # method 2: random generate based on the number of most common features
                    rand_nums = np.random.choice([0, 1], size=feature_dim, p=[.986, .014])
                    attr_matrix = np.append(attr_matrix, [rand_nums], axis=0)
                    _label = np.random.randint(7, size = 1)
                    labels = np.append(labels, _label, axis=0)

                # labels = np.append(labels, np.array([2]), axis=0)
                print(" attr matrix is", attr_matrix)

            dim_node = adj_matrix.shape[0] + self.added_node_num
            new_adj_matrix = np.ones(shape=(dim_node, dim_node))
            np.fill_diagonal(new_adj_matrix, 0)
            for i in range(adj_matrix.shape[0]):
                for j in range(adj_matrix.shape[0]):
                    if adj_matrix[i, j] == 0:
                        new_adj_matrix[i, j] = 0

        x = torch.from_numpy(attr_matrix).float()
        y = torch.from_numpy(labels).long()
        print(" ------ Final Shape ------")
        if self.added_node_num:
            print("adj shape: ", new_adj_matrix.shape, "att shape: ", attr_matrix.shape, "label shape: ", labels.shape)
            edge_index = dense_to_sparse(torch.from_numpy(new_adj_matrix))[0]
        else:
            print("adj shape: ", adj_matrix.shape, "att shape: ", attr_matrix.shape, "label shape: ", labels.shape)
            edge_index = dense_to_sparse(torch.from_numpy(adj_matrix))[0]
        data = Data(x=x, y=y, edge_index=edge_index)
        return data
