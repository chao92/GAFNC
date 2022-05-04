import os.path as osp

import torch
import tqdm
from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.io import read_planetoid_data
from torch_geometric.utils import dense_to_sparse, to_scipy_sparse_matrix
import utils
from scipy.sparse import csr_matrix
import numpy as np

class PlanetoidDataset(InMemoryDataset):
    r"""The citation network datasets "Cora", "CiteSeer" and "PubMed" from the
    `"Revisiting Semi-Supervised Learning with Graph Embeddings"
    <https://arxiv.org/abs/1603.08861>`_ paper.
    Nodes represent documents and edges represent citation links.
    Training, validation and test splits are given by binary masks.

    Args:
        root (string): Root directory where the dataset_preprocess should be saved.
        name (string): The name of the dataset_preprocess (:obj:`"Cora"`,
            :obj:`"CiteSeer"`, :obj:`"PubMed"`).
        split (string): The type of dataset_preprocess split
            (:obj:`"public"`, :obj:`"full"`, :obj:`"random"`).
            If set to :obj:`"public"`, the split will be the public fixed split
            from the
            `"Revisiting Semi-Supervised Learning with Graph Embeddings"
            <https://arxiv.org/abs/1603.08861>`_ paper.
            If set to :obj:`"full"`, all nodes except those in the validation
            and test sets will be used for training (as in the
            `"FastGCN: Fast Learning with Graph Convolutional Networks via
            Importance Sampling" <https://arxiv.org/abs/1801.10247>`_ paper).
            If set to :obj:`"random"`, train, validation, and test sets will be
            randomly generated, according to :obj:`num_train_per_class`,
            :obj:`num_val` and :obj:`num_test`. (default: :obj:`"public"`)
        num_train_per_class (int, optional): The number of training samples
            per class in case of :obj:`"random"` split. (default: :obj:`20`)
        num_val (int, optional): The number of validation samples in case of
            :obj:`"random"` split. (default: :obj:`500`)
        num_test (int, optional): The number of test samples in case of
            :obj:`"random"` split. (default: :obj:`1000`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    url = 'https://github.com/kimiyoung/planetoid/raw/master/data'

    def __init__(self, root, name, transform=None,
                 pre_transform=None, added_new_nodes=None):
        self.name = name.lower()
        self.train_index = None
        self.val_index = None
        self.test_index = None
        if added_new_nodes:
            self.added_node_num = added_new_nodes
        else:
            self.added_node_num = 0
        super(PlanetoidDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])


    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        # return osp.join(self.root, self.name, 'processed')
        if self.added_node_num:
            return osp.join(self.root, self.name, str(self.added_node_num), 'processed')
        else:
            return osp.join(self.root, self.name, 'processed')
    @property
    def raw_file_names(self):
        names = ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']
        return ['ind.{}.{}'.format(self.name.split('-')[0].lower(), name) for name in names]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        for name in self.raw_file_names:
            download_url('{}/{}'.format(self.url, name), self.raw_dir)

    def process(self):
        #
        data = self.preprocessing_cora_data()

        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def preprocessing_cora_data(self):
        """ Load a SparseGraph from a Numpy binary file.
        Returns:

        """
        print("self raw paths is", self.raw_paths[0])
        data = read_planetoid_data(self.raw_dir, self.name.split('-')[0])
        adj_matrix, attr_matrix, labels = to_scipy_sparse_matrix(
            data.edge_index), csr_matrix(data.x), data.y
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
                # method 1: random select from origin attr matrix
                rd_index = np.random.randint(attr_matrix.shape[0],size = 1)
                attr_matrix = np.append(attr_matrix, attr_matrix[rd_index], axis=0)
                labels = np.append(labels, labels[rd_index], axis=0)

                # method 2: random generate based on the number of most common features
                # rand_nums = np.random.choice([0, 1], size=feature_dim, p=[.986, .014])
                # attr_matrix = np.append(attr_matrix, [rand_nums], axis=0)
                # _label = np.random.randint(7, size = 1)
                # labels = np.append(labels, _label, axis=0)


                # labels = np.append(labels, np.array([2]), axis=0)
                print(" attr matrix is", attr_matrix)

            dim_node = adj_matrix.shape[0] + self.added_node_num
            new_adj_matrix = np.ones(shape=(dim_node, dim_node))
            np.fill_diagonal(new_adj_matrix, 0)
            for i in tqdm.tqdm(range(adj_matrix.shape[0])):
                for j in range(adj_matrix.shape[0]):
                    if adj_matrix[i, j] == 0:
                        new_adj_matrix[i, j] = 0

        x = torch.from_numpy(attr_matrix).float()
        if isinstance(labels, torch.Tensor):
            y = labels.long()
        else:
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


    def __repr__(self):
        return '{}()'.format(self.name)
