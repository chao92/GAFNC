import torch
from torch import Tensor
from torch_geometric.utils.loop import add_self_loops
from torch.nn.functional import cross_entropy
from attack.base_attack import AttackBase
EPS = 1e-15

def cross_entropy_with_logit(y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs):
    return cross_entropy(y_pred, y_true.long(), **kwargs)

class GNNAttack(AttackBase):
    r"""The GNN-Explainer model from the `"GNNExplainer: Generating
    Explanations for Graph Neural Networks"
    <https://arxiv.org/abs/1903.03894>`_ paper for identifying compact subgraph
    structures and small subsets node features that play a crucial role in a
    GNN’s node-predictions.

    .. note:: For an example, see `benchmarks/xgraph
        <https://github.com/divelab/DIG/tree/dig/benchmarks/xgraph>`_.

    Args:
        model (torch.nn.Module): The GNN module to attack.
        new_node_num (int): The number to added to graph
            (default: 0)
        epochs (int, optional): The number of epochs to train.
            (default: :obj:`100`)
        lr (float, optional): The learning rate to apply.
            (default: :obj:`0.01`)
        attack_graph (bool, optional): Whether to attack model globally
            (default: :obj:`True`)
    """

    coeffs = {
        'edge_size': 1e-5,
        'node_feat_size': 1e-5,
        'edge_ent': 1.0,
        'node_feat_ent': 1,
    }

    def __init__(self, model, new_node_num=0, epochs=100, lr=0.01, attack_graph=True, mask_features=False, mask_structure=True, indirect_level=0, random_structure=False, random_feature=False, \
                 args=None):
        if args:
            self.coeffs['edge_size'] = args.edge_size
            self.coeffs['edge_ent'] = args.edge_ent
            self.coeffs['node_feat_size'] = args.node_feat_size
            self.coeffs['node_feat_ent'] = args.node_feat_ent
            self.coeffs['node_id'] = args.node_idx
            self.coeffs['desired_class'] = args.desired_class
        super(GNNAttack, self).__init__(model, new_node_num, epochs, lr, attack_graph, mask_features, mask_structure, indirect_level, random_structure, random_feature)
        print(" indirect level is ------ ", self.indirect_level)


    def __loss__(self, raw_preds, x_labels):
        if self.attack_graph:
            print(" raw_preds", raw_preds.shape, "x labels", x_labels.shape)
            loss = cross_entropy_with_logit(raw_preds, x_labels)
        else:
            print("slef. node idx", self.coeffs['node_id'])
            # print("loss predict:",raw_preds[self.coeffs['node_id']].unsqueeze(0).shape, raw_preds[self.coeffs['node_id']].unsqueeze(0))
            # print(" x_labels", x_labels)
            # x_label = torch.tensor([x_labels[self.coeffs['node_id']]])
            # close to desired label
            x_label = torch.tensor([self.coeffs['desired_class']])
            # print(" teorch ", x_label)
            loss = cross_entropy_with_logit(raw_preds[self.coeffs['node_id']].unsqueeze(0), x_label.to(raw_preds.device))
            loss *= -0.1

        print(" cross entry ", loss)
        loss *= -10
        print(" cross entry time -1", loss)
        if self.mask_structure:
            print(" structure attack ", self.mask_structure)
            m = self.edge_mask.sigmoid()
            # print(" loss, edge mask is", m.shape, "and m is", m)
            print(" torch norm is", torch.norm(m, 1))
            l1_regularization = self.coeffs['edge_size'] * torch.norm(m, 1)
            loss += l1_regularization
            print("l1 is", l1_regularization)
            # loss = loss + self.coeffs['edge_size'] * m.sum()
            ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
            print(" edge ent ", ent.mean())
            loss = loss + self.coeffs['edge_ent'] * ent.mean()
            print(" loss ---- ", loss)
        if self.mask_features:
            print(" feature attack ", self.mask_features)
            m = self.node_feat_mask.sigmoid()
            print(" node feature size ", m.sum())
            loss = loss + self.coeffs['node_feat_size'] * m.sum()
            ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
            print(" node ent ", ent.mean())
            loss = loss + self.coeffs['node_feat_ent'] * ent.mean()
        print(" final loss is ", loss)
        return loss

    def gnn_explainer_alg(self,
                          x: Tensor,
                          edge_index: Tensor,
                          ex_label: Tensor,
                          **kwargs
                          ):

        # initialize a mask
        self.to(x.device)

        # train to get the mask
        optimizer = torch.optim.Adam([self.node_feat_mask, self.edge_mask],
                                     lr=self.lr)

        epsion = float("inf")
        prev_loss = None
        epoch = 0
        maximum = 3000
        _limit = 1e-5
        while epsion >= _limit and epoch<=maximum:
            if epoch % 50 == 0:
                print(" -------------------attack epoch = ", epoch)
            # print(" shape of mask of x is", self.node_feat_mask.sigmoid().shape)

            if self.mask_features:
                h1 = x[:-self.added_node_num]
                h2 = x[-self.added_node_num:] * self.node_feat_mask.sigmoid()

                h = torch.cat((h1,h2), 0)
                # print(" h1. shape", h1.shape, "h2.shape", h2.shape, "h shape", h.shape)
            else:
                h = x

            # print(" h shape is", h.shape)
            raw_preds = self.model(x=h, edge_index=edge_index, **kwargs)
            loss = self.__loss__(raw_preds, ex_label)
            if not prev_loss:
                prev_loss = loss
            else:
                epsion = abs(prev_loss - loss)
                prev_loss = loss
            if epoch % 20 == 0:
                print(" epsion :", epsion)
                print(f'attack Loss:{loss.item()}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch += 1

        print(" edge mask", self.edge_mask.data)
        print(" shape of node feat mask", self.node_feat_mask.shape)
        print(" data of node feat mask", self.node_feat_mask.data)
        return self.node_feat_mask.data, self.edge_mask.data

    def forward(self, x, edge_index, y, **kwargs):
        r"""
        Run the explainer for a specific graph instance.

        Args:
            x (torch.Tensor): The graph instance's input node features.
            edge_index (torch.Tensor): The graph instance's edge index.
            mask_features (bool, optional): Whether to use feature mask. Not recommended.
                (Default: :obj:`False`)
            **kwargs (dict):
                :obj:`node_idx` （int): The index of node that is pending to be explained.
                (for node classification)
                :obj:`sparsity` (float): The Sparsity we need to control to transform a
                soft mask to a hard mask. (Default: :obj:`0.7`)
                :obj:`num_classes` (int): The number of task's classes.

        :rtype: (None, list, list)

        .. note::
            (None, edge_masks, related_predictions):
            edge_masks is a list of edge-level explanation for each class;
            related_predictions is a list of dictionary for each class
            where each dictionary includes 4 type predicted probabilities.

        """
        super().forward(x=x, edge_index=edge_index, **kwargs)
        self.model.eval()

        self_loop_edge_index, _ = add_self_loops(edge_index, num_nodes=self.num_nodes)

        # Assume the mask we will predict
        # labels = tuple(i for i in range(kwargs.get('num_classes')))
        # ex_labels = tuple(torch.tensor([label]).to(self.device) for label in labels)
        # Calculate mask
        edge_masks = []
        feature_masks = []
        self.__clear_masks__()
        self.__set_masks__(x, self_loop_edge_index)

        # when random select mask is set as true skip the train step
        f_mask, e_mask = self.gnn_explainer_alg(x, edge_index, y)
        if self.random_mask_structure:
            e_mask = self.edge_mask.data.clone()
        if self.random_mask_feature:
            f_mask = self.node_feat_mask.data.clone()

        structure_sp = None
        feature_sp = None
        tmp_feature_mask, tmp_edge_mask,  structure_sp, feature_sp = self.control_sparsity(f_mask, e_mask, fix_sparsity = kwargs.get('fix_sparsity'), \
                                                                                           sparsity=kwargs.get('structure_sparsity'), feat_sparsity=kwargs.get('feat_sparsity'))
        edge_masks.append(tmp_edge_mask)
        feature_masks.append(tmp_feature_mask)

        # for ex_label in ex_labels:
        #     self.__clear_masks__()
        #     self.__set_masks__(x, self_loop_edge_index)
        #     edge_masks.append(self.control_sparsity(self.gnn_explainer_alg(x, edge_index, ex_label), sparsity=kwargs.get('sparsity')))
        #     # edge_masks.append(self.gnn_explainer_alg(x, edge_index, ex_label))

        # with torch.no_grad():
        #     related_preds = self.eval_related_pred(x, edge_index, edge_masks, **kwargs)
        self.__clear_masks__()
        return None, edge_masks, feature_masks, structure_sp, feature_sp
        # return None, edge_masks, feature_masks, related_preds

    def __repr__(self):
        return f'{self.__class__.__name__}()'
