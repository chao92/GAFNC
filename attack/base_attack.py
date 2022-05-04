from typing import List, Tuple, Dict
from math import sqrt
import torch
from torch import Tensor
import torch.nn as nn
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.nn import MessagePassing
from torch_geometric.utils.loop import add_self_loops
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
# from attack.models import subgraph
# from rdkit import Chem
from matplotlib.axes import Axes
from matplotlib.patches import Path, PathPatch
import collections
import numpy as np
from attack.models import GNNPool


EPS = 1e-15


class AttackBase(nn.Module):
    """
    for target attack
    indirect_level = 0: no constrain, newly added nodes can be directly connected to any nodes
    indirect_level = 1: with constrain that, no directly connected with target node itself
    indirect_level = 2: with constrain that, no directly connected with target node 1-hop neighbors
    indirect_level = 3: with constrain that, no directly connected with target node 2-hop neighbors
    """

    def __init__(self, model: nn.Module, new_node_num=0, epochs=0, lr=0, attack_graph=True, mask_features=False, mask_structure=True, indirect_level=0, random_structure=False, random_feature=False, molecule=False):
        super().__init__()
        self.model = model
        self.lr = lr
        self.epochs = epochs
        self.attack_graph = attack_graph
        self.mask_features = mask_features
        self.mask_structure = mask_structure
        self.random_feature = random_feature
        self.random_mask_structure = random_structure
        self.random_mask_feature = random_feature
        self.molecule = molecule
        self.mp_layers = [module for module in self.model.modules() if isinstance(module, MessagePassing)]
        self.num_layers = len(self.mp_layers)

        self.ori_pred = None
        self.ex_labels = None
        self.edge_mask = None
        self.updated_edge_mask = None
        self.fixed_edge_mask = None

        self.hard_edge_mask = None

        self.added_node_num = new_node_num
        self.indirect_level = indirect_level

        self.num_edges = None
        self.num_nodes = None
        self.device = None
        # self.table = Chem.GetPeriodicTable().GetElementSymbol

    def __construct_mask__(self, edge_index):
        mask_size = edge_index.shape[1]
        # print(" edge index is", edge_index)
        # print(" edge index [0][0] is", edge_index[0][0])
        # print(" edge index [0][1] is", edge_index[0][1])
        # print(" edge index [1][0] is", edge_index[1][0])
        """
        start add indirectly target attak
        """

        # TODO
        # node_idx = 341
        # banned_nodes = []
        # def ___get_hop_k_neighbors___(edge_index, node_idx, k):
        #     node_list = collections.deque([node_idx])
        #     while node_list and k:
        #         for i in range(len(node_list)):
        #             node_list.popleft()
        #     return node_list
        #
        # def __get_neighbors__(edge_index, node_idx):
        #     neighbors = []
        #     cord_x, cord_y = edge_index[0], edge_index[1]
        #     idx = (cord_x == node_idx).nonzero(as_tuple=True)[0]
        #     idy = (cord_y == node_idx).nonzero(as_tuple=True)[0]
        #     for item in idx:
        #         neighbors.append(edge_index[1][item].item())
        #         # print(" idx cor: ", edge_index[1][item])
        #     for item in idy:
        #         neighbors.append(edge_index[0][item].item())
        #         # print(" idy cor ", edge_index[0][item])
        #     # print("type ",type(neighbors))
        #     # print("neigh", neighbors)
        #     # print(" set is", set(neighbors))
        #     return list(set(neighbors))
        # __get_neighbors__(edge_index, node_idx)
        # exit(-2)
        # not directly connect to target node
        # if self.indirect_level == 1:
        #     banned_nodes.append(node_idx)
        # # not directly connect to target node and target node 1-hop neighbors
        # elif self.indirect_level == 2:
        #     banned_nodes.append()
        # # not directly connect to target node and target node 1-hop and 1-hop neighbors
        # elif self.indirect_level == 3:
        #     banned_nodes.append()
        """
        end add indirectly target attak
        """


        cord_x = edge_index[0]
        cord_y = edge_index[1]
        updated_mask = None
        print(" num_node =",self.num_nodes)
        print(" new added ", self.added_node_num)
        for i in range(self.num_nodes-self.added_node_num, self.num_nodes):
            idx = (cord_x == i).nonzero(as_tuple=True)[0]
            idy = (cord_y == i).nonzero(as_tuple=True)[0]
            if updated_mask == None:
                updated_mask = torch.cat((idx, idy))
                # updated_mask = idx
            else:
                updated_mask = torch.cat((updated_mask, idx))
                updated_mask = torch.cat((updated_mask, idy))
        # updated_mask = torch.unique(ids, sorted=True)
        # print("size of update = ",len(torch.unique(updated_mask)))
        updated_mask = torch.unique(updated_mask)
        updated_mask_size = updated_mask.shape[0]
        # edge_mask = torch.randn(mask_size)
        edge_mask = torch.randn(mask_size)
        std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * self.num_nodes))
        initialize_update_mask = torch.randn(updated_mask_size, requires_grad=True)*std
        initialize_fixed_mask = torch.ones(mask_size - updated_mask_size, requires_grad=False)
        edge_mask[updated_mask] = initialize_update_mask
        fixed_mask = []
        for i in range(mask_size):
            if i not in updated_mask:
                fixed_mask.append(i)
        print(" len(fixed)", len(fixed_mask), "updated = ", updated_mask_size, "mask size =", mask_size)
        assert len(fixed_mask) + updated_mask_size == mask_size
        fixed_mask = torch.LongTensor(fixed_mask)
        edge_mask[fixed_mask] = initialize_fixed_mask

        self.updated_edge_mask = updated_mask
        self.fixed_edge_mask = fixed_mask

        print(" total mask size ", mask_size, "fixed mask size :", fixed_mask.shape[0], " updated mask size: ", updated_mask_size)
        return edge_mask

    def __set_masks__(self, x, edge_index, init="normal"):
        (N, F), E = x.size(), edge_index.size(1)

        print(" node = ", N, " feature dim = ", F, " edge size = ", E)

        edge_mask = self.__construct_mask__(edge_index)

        std = 0.1
        self.node_feat_mask = torch.nn.Parameter(torch.randn((self.added_node_num,F), requires_grad=True, device=self.device) * 0.1)
        std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
        # self.edge_mask = torch.nn.Parameter(torch.randn(E, requires_grad=True, device=self.device) * std)
        # self.edge_mask = torch.nn.Parameter(100 * torch.ones(E, requires_grad=True))
        self.edge_mask = torch.nn.Parameter(edge_mask)

        for module in self.model.modules():
            # print(" module is", module)
            if isinstance(module, MessagePassing):
                # print(" instance of MP ", module)
                module.__explain__ = True
                module.__edge_mask__ = self.edge_mask

    def __clear_masks__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False
                module.__edge_mask__ = None
        self.node_feat_masks = None
        self.edge_mask = None

    @property
    def __num_hops__(self):
        if self.attack_graph:
            return -1
        else:
            return self.num_layers

    def __flow__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                return module.flow
        return 'source_to_target'

    # def __subgraph__(self, node_idx, x, edge_index, **kwargs):
    #     num_nodes, num_edges = x.size(0), edge_index.size(1)
    #
    #     subset, edge_index, mapping, edge_mask = subgraph(
    #         node_idx, self.__num_hops__, edge_index, relabel_nodes=True,
    #         num_nodes=num_nodes, flow=self.__flow__())
    #
    #     x = x[subset]
    #     for key, item in kwargs.items():
    #         if torch.is_tensor(item) and item.size(0) == num_nodes:
    #             item = item[subset]
    #         elif torch.is_tensor(item) and item.size(0) == num_edges:
    #             item = item[edge_mask]
    #         kwargs[key] = item
    #
    #     return x, edge_index, mapping, edge_mask, kwargs


    def forward(self,
                x: Tensor,
                edge_index: Tensor,
                **kwargs
                ):
        self.num_edges = edge_index.shape[1]
        self.num_nodes = x.shape[0]
        self.device = x.device


    def control_sparsity(self, feat_mask, mask, fix_sparsity=None, sparsity=None, feat_sparsity=None, **kwargs):
        r"""

        :param mask: mask that need to transform
        :param sparsity: sparsity we need to control i.e. 0.7, 0.5
        :return: transformed mask where top 1 - sparsity values are set to inf.
        """
        if sparsity is None:
            sparsity = 0.7
        if feat_sparsity is None:
            feat_sparsity = 0.7

        print(" structure sparsity is ", sparsity, " feature sparsity is ", feat_sparsity)

        mask_len = self.updated_edge_mask.shape[0]
        split_point = int((1 - sparsity) * mask_len)
        trans_mask = mask.clone()

        structure_sp = None
        feature_sp = None
        print(" total mask len is", len(trans_mask))

        # For structure mask
        trans_mask[:] = -float('inf')
        print(" updated mask len = ", mask_len, "split poitn = ", split_point)
        if self.random_mask_structure:
            for idx in self.fixed_edge_mask:
                trans_mask[idx] = float('inf')
            indices = torch.randperm(mask_len)[:split_point]
            for idx in self.updated_edge_mask[indices]:
                trans_mask[idx] = float('inf')
            structure_sp = sparsity
        else:
            sorted_mask, indices = torch.sort(mask, descending=True)
            print(" sorted mask is", sorted_mask)

            cnt = 0
            print(" fix sparisty is", fix_sparsity)
            if not fix_sparsity:
                for idx in indices:
                    if idx in self.fixed_edge_mask:
                        trans_mask[idx] = float('inf')
                    elif idx in self.updated_edge_mask:
                        if mask[idx] > 0:
                            trans_mask[idx] = float('inf')
                            cnt+=1
            else:
                print(" fixed sparsity-----------------")
                for idx in indices:
                    if idx in self.fixed_edge_mask.to(idx.device):
                        trans_mask[idx] = float('inf')
                    elif idx in self.updated_edge_mask.to(idx.device) and cnt <= split_point:
                        trans_mask[idx] = float('inf')
                        cnt += 1

            structure_sp = cnt/mask_len
            print(" non zero cnt is ", cnt, "structure sparsity = ", structure_sp)

        # For feature mask
        trans_feat_mask = None
        if self.mask_features:
            rows, cols = feat_mask.shape[0], feat_mask.shape[1]
            trans_feat_mask = feat_mask.clone()
            split_point = int((1 - feat_sparsity) * cols)
            print(" rows, cols", rows, cols)
            print(" split point = ", split_point, trans_feat_mask.shape)
            if self.random_mask_feature:
                print("------------------------- random feature mask-------------------------")
                for i in range(rows):
                    indices = torch.randperm(cols)[:split_point]
                    for idx in range(cols):
                        if idx in indices:
                            trans_feat_mask[i][idx] = 1
                        else:
                            trans_feat_mask[i][idx] = 0
                feature_sp = feat_sparsity
            else:
                sorted_feat_mask, indices = torch.sort(feat_mask, descending=True)
                print(" sorted feature mask",sorted_feat_mask)
                non_zero_cnt = 0
                if not fix_sparsity:
                    for i in range(rows):
                        for j in range(cols):
                            if trans_feat_mask[i][indices[i][j]] > 0:
                                trans_feat_mask[i][indices[i][j]] = 1
                                non_zero_cnt += 1
                            else:
                                trans_feat_mask[i][indices[i][j]] = 0
                    feature_sp = non_zero_cnt/(rows*cols)
                else:
                    for i in range(rows):
                        for j in range(cols):
                            if j <= split_point:
                                trans_feat_mask[i][indices[i][j]] = 1
                                non_zero_cnt += 1
                            else:
                                trans_feat_mask[i][indices[i][j]] = 0
                    feature_sp = non_zero_cnt / (rows*cols)
                print(" non zero cnt is ", non_zero_cnt, "feature sparsity = ", feature_sp)
        else:
            feature_sp = feat_sparsity

        return trans_feat_mask, trans_mask, structure_sp, feature_sp



# class WalkBase(AttackBase):
#
#     def __init__(self, model: nn.Module, epochs=0, lr=0, explain_graph=False, molecule=False):
#         super().__init__(model, epochs, lr, explain_graph, molecule)
#
#     def extract_step(self, x, edge_index, detach=True, split_fc=False):
#
#         layer_extractor = []
#         hooks = []
#
#         def register_hook(module: nn.Module):
#             if not list(module.children()) or isinstance(module, MessagePassing):
#                 hooks.append(module.register_forward_hook(forward_hook))
#
#         def forward_hook(module: nn.Module, input: Tuple[Tensor], output: Tensor):
#             # input contains x and edge_index
#             if detach:
#                 layer_extractor.append((module, input[0].clone().detach(), output.clone().detach()))
#             else:
#                 layer_extractor.append((module, input[0], output))
#
#         # --- register hooks ---
#         self.model.apply(register_hook)
#
#         pred = self.model(x, edge_index)
#
#         for hook in hooks:
#             hook.remove()
#
#         # --- divide layer sets ---
#
#         walk_steps = []
#         fc_steps = []
#         pool_flag = False
#         step = {'input': None, 'module': [], 'output': None}
#         for layer in layer_extractor:
#             if isinstance(layer[0], MessagePassing) or isinstance(layer[0], GNNPool):
#                 if isinstance(layer[0], GNNPool):
#                     pool_flag = True
#                 if step['module'] and step['input'] is not None:
#                     walk_steps.append(step)
#                 step = {'input': layer[1], 'module': [], 'output': None}
#             if pool_flag and split_fc and isinstance(layer[0], nn.Linear):
#                 if step['module']:
#                     fc_steps.append(step)
#                 step = {'input': layer[1], 'module': [], 'output': None}
#             step['module'].append(layer[0])
#             step['output'] = layer[2]
#
#
#         for walk_step in walk_steps:
#             if hasattr(walk_step['module'][0], 'nn') and walk_step['module'][0].nn is not None:
#                 # We don't allow any outside nn during message flow process in GINs
#                 walk_step['module'] = [walk_step['module'][0]]
#
#
#         if split_fc:
#             if step['module']:
#                 fc_steps.append(step)
#             return walk_steps, fc_steps
#         else:
#             fc_step = step
#
#
#         return walk_steps, fc_step
#
#     def walks_pick(self,
#                    edge_index: Tensor,
#                    pick_edge_indices: List,
#                    walk_indices: List=[],
#                    num_layers=0
#                    ):
#         walk_indices_list = []
#         for edge_idx in pick_edge_indices:
#
#             # Adding one edge
#             walk_indices.append(edge_idx)
#             _, new_src = src, tgt = edge_index[:, edge_idx]
#             next_edge_indices = np.array((edge_index[0, :] == new_src).nonzero().view(-1))
#
#             # Finding next edge
#             if len(walk_indices) >= num_layers:
#                 # return one walk
#                 walk_indices_list.append(walk_indices.copy())
#             else:
#                 walk_indices_list += self.walks_pick(edge_index, next_edge_indices, walk_indices, num_layers)
#
#             # remove the last edge
#             walk_indices.pop(-1)
#
#         return walk_indices_list
#
#     def eval_related_pred(self, x, edge_index, masks, **kwargs):
#
#         node_idx = kwargs.get('node_idx')
#         node_idx = 0 if node_idx is None else node_idx # graph level: 0, node level: node_idx
#
#         related_preds = []
#
#         for label, mask in enumerate(masks):
#             # origin pred
#             for edge_mask in self.edge_mask:
#                 edge_mask.data = float('inf') * torch.ones(mask.size(), device=self.device)
#             ori_pred = self.model(x=x, edge_index=edge_index, **kwargs)
#
#             for edge_mask in self.edge_mask:
#                 edge_mask.data = mask
#             masked_pred = self.model(x=x, edge_index=edge_index, **kwargs)
#
#             # mask out important elements for fidelity calculation
#             for edge_mask in self.edge_mask:
#                 edge_mask.data = - mask
#             maskout_pred = self.model(x=x, edge_index=edge_index, **kwargs)
#
#             # zero_mask
#             for edge_mask in self.edge_mask:
#                 edge_mask.data = - float('inf') * torch.ones(mask.size(), device=self.device)
#             zero_mask_pred = self.model(x=x, edge_index=edge_index, **kwargs)
#
#             # Store related predictions for further evaluation.
#             related_preds.append({'zero': zero_mask_pred[node_idx],
#                                   'masked': masked_pred[node_idx],
#                                   'maskout': maskout_pred[node_idx],
#                                   'origin': ori_pred[node_idx]})
#
#             # Adding proper activation function to the models' outputs.
#             related_preds[label] = {key: pred.softmax(0)[label].item()
#                                     for key, pred in related_preds[label].items()}
#
#         return related_preds
#
#
#     def explain_edges_with_loop(self, x: Tensor, walks: Dict[Tensor, Tensor], ex_label):
#
#         walks_ids = walks['ids']
#         walks_score = walks['score'][:walks_ids.shape[0], ex_label].reshape(-1)
#         idx_ensemble = torch.cat([(walks_ids == i).int().sum(dim=1).unsqueeze(0) for i in range(self.num_edges + self.num_nodes)], dim=0)
#         hard_edge_attr_mask = (idx_ensemble.sum(1) > 0).long()
#         hard_edge_attr_mask_value = torch.tensor([float('inf'), 0], dtype=torch.float, device=self.device)[hard_edge_attr_mask]
#         edge_attr = (idx_ensemble * (walks_score.unsqueeze(0))).sum(1)
#         # idx_ensemble1 = torch.cat(
#         #     [(walks_ids == i).int().sum(dim=1).unsqueeze(1) for i in range(self.num_edges + self.num_nodes)], dim=1)
#         # edge_attr1 = (idx_ensemble1 * (walks_score.unsqueeze(1))).sum(0)
#
#         return edge_attr - hard_edge_attr_mask_value
#
#     class connect_mask(object):
#
#         def __init__(self, cls):
#             self.cls = cls
#
#         def __enter__(self):
#
#             self.cls.edge_mask = [nn.Parameter(torch.randn(self.cls.x_batch_size * (self.cls.num_edges + self.cls.num_nodes))) for _ in
#                              range(self.cls.num_layers)] if hasattr(self.cls, 'x_batch_size') else \
#                                  [nn.Parameter(torch.randn(1 * (self.cls.num_edges + self.cls.num_nodes))) for _ in
#                              range(self.cls.num_layers)]
#
#             for idx, module in enumerate(self.cls.mp_layers):
#                 module.__explain__ = True
#                 module.__edge_mask__ = self.cls.edge_mask[idx]
#
#         def __exit__(self, *args):
#             for idx, module in enumerate(self.cls.mp_layers):
#                 module.__explain__ = False
