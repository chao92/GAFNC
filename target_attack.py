import argparse
import sys
import os.path as osp
import os
sys.path.insert(1, osp.abspath(osp.join(os.getcwd(), *('..',)*2)))
from dataset_preprocess import CoraDataset, PlanetoidDataset
from attack.models import *
import torch
import pandas as pd
from tqdm.notebook import tqdm
from attack.GAFNC import GNNAttack
from torch_geometric.utils.loop import add_self_loops, remove_self_loops
import utils
import numpy as np
import pickle

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask

def split_dataset(dataset, new_nodes, train_percent=0.7):
    indices = []
    _size = dataset.data.num_nodes - new_nodes
    y = dataset.data.y[:_size]
    for i in range(dataset.num_classes):
        index = (y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:int(len(i) * train_percent)] for i in indices], dim=0)
    rest_index = torch.cat([i[int(len(i) * train_percent):] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    dataset.data.train_mask = index_to_mask(train_index, size=dataset.data.num_nodes)
    dataset.data.val_mask = index_to_mask(rest_index[:len(rest_index) // 2], size=dataset.data.num_nodes)
    dataset.data.test_mask = index_to_mask(rest_index[len(rest_index) // 2:], size=dataset.data.num_nodes)

    dataset.train_index = train_index[:]
    dataset.val_index = rest_index[:len(rest_index) // 2]
    dataset.test_index = rest_index[len(rest_index) // 2:]

    dataset.data, dataset.slices = dataset.collate([dataset.data])

    return dataset

def build_args():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='cora', help='name of dataset_preprocess')
    # dataset_name = ['cora', 'citeseer', 'pubmed']
    parser.add_argument('--attack_graph', type=str2bool, default=True, help='global attack')
    parser.add_argument('--node_idx', type=int, default=None, help='no target idx')
    parser.add_argument('--structure_attack', type=str2bool, default=True, help='with structure attack')
    parser.add_argument('--feature_attack', type=str2bool, default=False, help='with feature attack')
    parser.add_argument('--added_node_num', type=int, default=20, help='num of new nodes')
    parser.add_argument('--train_percent', type=float, default=0.7, help='train percent')
    parser.add_argument('--fix_sparsity', type=str2bool, default=True, help='control the attack sparsity')
    parser.add_argument('--sparsity', type=float, default=0.5, help='sparsity')
    parser.add_argument('--feat_sparsity', type=float, default=0.5, help='feat_sparsity')
    parser.add_argument('--random_structure', type=str2bool, default=False, help='random mask')
    parser.add_argument('--random_feature', type=str2bool, default=False, help='random mask of feature')
    parser.add_argument('--edge_size', type=float, default=1e-5, help='edge_size')
    parser.add_argument('--edge_ent', type=float, default=1.0, help='edge_ent')
    parser.add_argument('--node_feat_size', type=float, default=1e-5, help='edge_size')
    parser.add_argument('--node_feat_ent', type=float, default=1.0, help='edge_ent')
    parser.add_argument('--train_epochs', type=int, default=300, help='epochs for training a GNN model')
    parser.add_argument('--attack_epochs', type=int, default=600, help='epochs for attacking a GNN model')
    parser.add_argument('--retrain_epochs', type=int, default=10,
                        help='epochs for retraining a GNN model with new graph')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--desired_class', type=int, default=None, help='attack specific node to desired class')

    parser.add_argument('--model_name', type=str, default="baseline", help='model variants name')

    parser.add_argument('--indirect_level', type=int, default=0, help='target indirect attack level')

    args = parser.parse_args()
    return args

def fix_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi gpu
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    np.random.seed(seed)

def eval_all(model, data):
    train_loss, train_acc = utils.evaluate(model, data, data.train_mask)
    val_loss, val_acc = utils.evaluate(model, data, data.val_mask)
    test_loss, test_acc = utils.evaluate(model, data, data.test_mask)
    return [train_loss, test_loss, val_loss, train_acc, test_acc, val_acc]

if __name__ == '__main__':

    args = build_args()

    print("args", args)
    fix_random_seed(seed=args.seed)
    ADD_ZERO = 0
    # step 1: load baseline dataset_preprocess
    data_name = args.dataset_name
    if data_name in ["cora", 'photo']:
        baseline = CoraDataset('./datasets', data_name, added_new_nodes=ADD_ZERO)
    else:
        # for dataset_preprocess pubmed, and citeseer
        baseline = PlanetoidDataset('./datasets', data_name, added_new_nodes=ADD_ZERO)

    split_dataset_name = "baseline_"+data_name+"_split"
    split_path = osp.join('./datasets', split_dataset_name, 'train_percent', str(args.train_percent), 'added_node', str(ADD_ZERO))
    if not osp.isdir(split_path):
        dataset = split_dataset(baseline, ADD_ZERO, train_percent=args.train_percent)
        os.makedirs(split_path)
        torch.save(baseline, osp.join(split_path, 'split_data.pt'))
    else:
        baseline = torch.load(osp.join(split_path, 'split_data.pt'))

    dim_node = baseline.num_node_features
    dim_edge = baseline.num_edge_features
    num_classes = baseline.num_classes
    baseline_model_ckpt_path = osp.join('checkpoints', data_name, str(args.train_percent), 'GCN_2l', 'seed', '0', 'GCN_2l_best.ckpt')

    # step 2: attack
    # add new nodes to origin dataset_preprocess
    added_node_num = args.added_node_num
    added_data_name = data_name + "-added"
    if data_name in ["cora", 'photo']:
        added_dataset = CoraDataset('./datasets', added_data_name, added_new_nodes=added_node_num)
    else:
        added_dataset = PlanetoidDataset('./datasets', added_data_name, added_new_nodes=added_node_num)

    if args.feature_attack:
        added_dataset.data.x[-added_node_num:] = 1
        print("feature attack ", added_dataset.data.x[-added_node_num:])

    added_indices = torch.as_tensor(list(range(baseline.data.num_nodes, baseline.data.num_nodes+added_node_num)))
    add_train_index = torch.cat((baseline.train_index, added_indices), dim=0)
    added_dataset.data.train_mask = index_to_mask(add_train_index, size=added_dataset.data.num_nodes)

    added_dataset.data.val_mask = index_to_mask(baseline.val_index, size=added_dataset.data.num_nodes)
    added_dataset.data.test_mask = index_to_mask(baseline.test_index, size=added_dataset.data.num_nodes)
    added_dataset.data, added_dataset.slices = added_dataset.collate([added_dataset.data])

    # step 2.1: load model
    print(" step 2.1: loading base model for attack")
    model = GCN_2l(model_level='node', dim_node=dim_node, dim_hidden=16, num_classes=num_classes)
    model.to(device)
    model.load_state_dict(torch.load(baseline_model_ckpt_path, map_location=device)['state_dict'])

    # step 2.2 attack
    attack_graph = args.attack_graph
    if attack_graph:
        print(" args.structure_attack", args.structure_attack)
        attacker = GNNAttack(model, new_node_num=added_node_num, epochs=args.attack_epochs, lr=0.005, attack_graph=attack_graph,
                         mask_features=args.feature_attack, mask_structure=args.structure_attack, indirect_level=args.indirect_level, random_structure=args.random_structure, random_feature=args.random_feature, args=args)
    else:
        # print(" random choise one id from test part of the datasete")
        # print(" test index is", baseline.test_index[0])
        # args.node_idx =
        print(" node idx is", args.node_idx)
        # args.node_idx = baseline.test_index[0].item()
        origin_label = baseline.data.y[args.node_idx]
        # args.desired_class = 2

        print(" target id is ", args.node_idx, " origin label is", origin_label, "desired label is ", args.desired_class)

        if args.node_idx == None and args.desired_class == None:
            print(" target attack, please input your target node id, and desired class id")
            exit(-1)
        attacker = GNNAttack(model, new_node_num=added_node_num, epochs=args.attack_epochs, lr=0.005, attack_graph=attack_graph,
                         mask_features=args.feature_attack,  mask_structure=args.structure_attack, indirect_level=args.indirect_level, random_structure=args.random_structure, random_feature=args.random_feature, args=args)
    attacker.to(device)
    sparsity = args.sparsity
    feat_sparsity = args.feat_sparsity
    fix_sparsity = args.fix_sparsity
    data = added_dataset.data
    data.to(device)
    print(" input file args is",args)
    walks, structure_masks, feature_masks, structure_sp, feature_sp = attacker(data.x, data.edge_index, data.y, fix_sparsity= fix_sparsity,sparsity=sparsity,feat_sparsity=feat_sparsity,\
                                                                               num_classes=num_classes)

    print(" strucutre sparisty =", structure_sp, " feature sparsity = ", feature_sp)

    # check train dataset predict shift
    # model.eval()
    # tmp_list = []
    # with torch.no_grad():
    #     output = model(baseline.data.x, baseline.data.edge_index, None)
    #     pred_class = torch.argmax(output[args.node_idx], dim=0).item()
    #     path = f'results/{data_name}/target_attack/added_node_{added_node_num}/train_percent_{args.train_percent}/desired_class_{args.desired_class}'
    #     if not osp.isdir(path):
    #         os.makedirs(path)
    #     print(" pred class is, ", pred_class)
    #     file = f'{path}/train_model_res.csv'
    #     cols=["ID", "desired_class", "pred_class", "pred_score"]
    #     tmp_list.append([args.node_idx, args.desired_class, pred_class, output[args.node_idx]])
    #     df = pd.DataFrame(tmp_list, columns=cols)
    #     if not os.path.isfile(file):
    #         df.to_csv(file, index=False)
    #     else:
    #         prev_res = pd.read_csv(file)
    #         final_res = pd.concat([df, prev_res],ignore_index=True)
    #         final_res.reset_index()
    #         final_res.to_csv(file, index=False)
    #     exit(-2)


    # step 2.3 apply learned mask to added_dataset
    # step 2.3.1 apply structure mask to dataset_preprocess
    print("mask dim", added_dataset.data.num_nodes)
    print(" edge index", added_dataset.data.edge_index.shape)
    print(" structur mask is", structure_masks)
    filter_indices = (structure_masks[0] == float('inf')).nonzero(as_tuple=True)[0]
    print(" filter indices = ", filter_indices)
    print(" filter indeices", filter_indices.shape)
    edge_index_with_loop, _ = add_self_loops(added_dataset.data.edge_index, num_nodes=added_dataset.data.num_nodes)
    added_dataset.data.edge_index = edge_index_with_loop
    print("dataset_preprocess.data.edge_index", added_dataset.data.edge_index.shape)
    added_dataset.data.edge_index = torch.index_select(added_dataset.data.edge_index, 1, filter_indices.to(device))
    print("after filter dataset_preprocess.data.edge_index", added_dataset.data.edge_index.shape)

    # step 2.3.2 apply feature mask to added_dataset
    if attacker.mask_features:
        added_dataset.data.x[-added_node_num:] *= feature_masks[0]

    # step 3: retrain model in changed dataset_preprocess
    del model
    model = GCN_2l(model_level='node', dim_node=added_dataset.num_node_features, dim_hidden=16,num_classes=added_dataset.num_classes)
    model.to(device)
    attack_ckpt_fold = osp.join('attack_checkpoints', data_name, str(added_node_num), 'GCN_2l')
    if not osp.isdir(attack_ckpt_fold):
        os.makedirs(attack_ckpt_fold)
    attack_ckpt_path = osp.join(attack_ckpt_fold, 'GCN_2l_best.ckpt')
    utils.train(model, added_dataset.data, attack_ckpt_path, lr=0.005, epochs=args.train_epochs,verbose=True)
    # [_, _, _, train_acc, test_acc, val_acc] = eval_all(model, added_dataset.data)


    if not args.attack_graph:

        path = f'results/target_attack/{data_name}/added_node_{added_node_num}/train_percent_{args.train_percent}/desired_class_{args.desired_class}'
        if not osp.isdir(path):
            os.makedirs(path)
        model.eval()
        with torch.no_grad():
            output = model(added_dataset.data.x, added_dataset.data.edge_index, None)
        success = None
        print(" node idx = ", args.node_idx)
        print(" output shape ", output.shape, output[args.node_idx], type(output[args.node_idx]))
        pred_class = torch.argmax(output[args.node_idx], dim=0).item()
        print(" pred class = ", pred_class)
        origin = added_dataset.data.y[args.node_idx]
        print(" origin", origin, "desired class", args.desired_class)
        cols = ["id", "pred_class", "desired_class", "success", "vis_path", "structure_sp", "feature_sp", "pred_score"]
        tmp_list = []
        vis_file = None
        if pred_class == args.desired_class:
            success = True
            vis_file = f'{path}/target_attack_{str(args.node_idx)}_{str(structure_sp)}_feature_sparsity_{str(feature_sp)}_dataset.pkl'
            utils.save_to_file([added_dataset.data.edge_index.to('cpu'), torch.argmax(output.to('cpu'), dim=1), added_dataset.data.x[:]], vis_file)
            baseline_fold = osp.join('./results/target_attack', data_name)
            baseline_vis_file = f'{baseline_fold}/train_percent_{args.train_percent}_baseline_A_X_res.pkl'


            # plot 1-hop and 2-hop figures center by node id
            with open(baseline_vis_file, 'rb') as f:
                edge_indx, pred, att = pickle.load(f)

            utils.viz_k_hop_op(edge_indx, pred, args.node_idx, 1, path, f'origin_center_node_{str(args.node_idx)}_hops_{str(1)}')
            utils.viz_k_hop_op(edge_indx, pred, args.node_idx, 2, path, f'origin_center_node_{str(args.node_idx)}_hops_{str(2)}')

            with open(vis_file, 'rb') as f:
                attack_edge_indx, attack_pred, attack_att = pickle.load(f)
            utils.viz_k_hop_op(attack_edge_indx, attack_pred, args.node_idx, 1, path,
                               f'attack_center_node_{str(args.node_idx)}_hops_{str(1)}')
            utils.viz_k_hop_op(attack_edge_indx, attack_pred, args.node_idx, 2, path,
                               f'attack_center_node_{str(args.node_idx)}_hops_{str(2)}')

        else:
            success = False

        tmp_list.append([args.node_idx, pred_class, args.desired_class, success, vis_file, structure_sp, feature_sp, output[args.node_idx]])
        df = pd.DataFrame(tmp_list,columns=cols)
        file = f'{path}/res.csv'
        if not os.path.isfile(file):
            df.to_csv(file, index=False)
        else:
            prev_res = pd.read_csv(file)
            final_res = pd.concat([df, prev_res],ignore_index=True)
            final_res.reset_index()
            final_res.to_csv(file, index=False)