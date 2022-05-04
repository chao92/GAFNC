import argparse
import sys
import os.path as osp
import os

sys.path.insert(1, osp.abspath(osp.join(os.getcwd(), *('..',) * 2)))
from dataset_preprocess import CoraDataset, PlanetoidDataset
from attack.models import *
import torch
import pandas as pd
from torch_geometric.utils.loop import add_self_loops
import utils
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def split_dataset(dataset, new_nodes, train_percent=0.7):
    torch.manual_seed(0)
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
    parser.add_argument('--added_node_num', type=int, default=20, help='num of new nodes')
    parser.add_argument('--train_percent', type=float, default=0.7, help='train percent')
    parser.add_argument('--sparsity', type=float, default=0.5, help='sparsity')
    parser.add_argument('--random', type=bool, default=False, help='random mask')
    parser.add_argument('--edge_size', type=float, default=0.005, help='edge_size')
    parser.add_argument('--edge_ent', type=float, default=1.0, help='edge_ent')
    parser.add_argument('--feat_sparsity', type=float, default=0.5, help='feat_sparsity')
    parser.add_argument('--train_epochs', type=int, default=300, help='epochs for training a GNN model')
    parser.add_argument('--attack_epochs', type=int, default=600, help='epochs for attacking a GNN model')
    parser.add_argument('--retrain_epochs', type=int, default=10,
                        help='epochs for retraining a GNN model with new graph')
    parser.add_argument('--seed', type=int, default=42, help='seed')

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


if __name__ == '__main__':

    args = build_args()
    fix_random_seed(seed=args.seed)
    ADD_ZERO = 0

    # step 1: train baseline
    data_name = args.dataset_name
    if data_name in ["cora", 'photo']:
        baseline = CoraDataset('./datasets', data_name, added_new_nodes=ADD_ZERO)
    else:
        # for dataset_preprocess pubmed, and citeseer
        baseline = PlanetoidDataset('./datasets', data_name, added_new_nodes=ADD_ZERO)

    print(" load data finished")
    split_dataset_name = "baseline_"+data_name+"_split"
    split_path = osp.join('./datasets', split_dataset_name, 'train_percent', str(args.train_percent), 'added_node', str(ADD_ZERO))
    if not osp.isdir(split_path):
        dataset = split_dataset(baseline, ADD_ZERO, train_percent=args.train_percent)
        os.makedirs(split_path)
        torch.save(baseline, osp.join(split_path, 'split_data.pt'))
    else:
        baseline = torch.load(osp.join(split_path, 'split_data.pt'))

    edge_index_with_loop, _ = add_self_loops(baseline.data.edge_index, num_nodes=baseline.data.num_nodes)
    baseline.data.edge_index = edge_index_with_loop
    dim_node = baseline.num_node_features
    dim_edge = baseline.num_edge_features

    num_classes = baseline.num_classes
    baseline_model_ckpt_fold = osp.join('checkpoints', data_name, str(args.train_percent),'GCN_2l', 'seed', str(args.seed))
    if not osp.isdir(baseline_model_ckpt_fold):
        os.makedirs(baseline_model_ckpt_fold)
    baseline_model_ckpt_path = osp.join(baseline_model_ckpt_fold,'GCN_2l_best.ckpt')


    model = GCN_2l(model_level='node', dim_node=dim_node, dim_hidden=16, num_classes=num_classes)
    model.to(device)


    if osp.isfile(baseline_model_ckpt_path):
        print(" loding from file")
        model.load_state_dict(torch.load(baseline_model_ckpt_path)['state_dict'])
    else:
        baseline.data = baseline.data.to(device)
        utils.train(model, baseline.data, baseline_model_ckpt_path, lr=0.005, epochs=args.train_epochs, verbose=False)

    # check baseline output of sepcific node for target attack
    if not args.attack_graph:
        print(" generate target attack test node list")
        model.eval()
        corrected_test_node_id_list = []
        cnt = 0
        with torch.no_grad():
            # print("baseline test index", baseline.test_index)
            output = model(baseline.data.x, baseline.data.edge_index, None)
            cols = ["test_ID", "predict_class"]
            success = None
            # y is from [0-6]
            tmp_list = []
            for id in baseline.test_index:
                # print("test id is ", id.item())
                node_idx = id.item()
                predict_class = torch.argmax(output[node_idx], dim=0).item()
                origin = baseline.data.y[node_idx]
                if origin == predict_class:
                    corrected_test_node_id_list.append(node_idx)
                    cnt += 1
                    # tmp_list.append([node_idx, predict_class, str(output[node_idx].detach().numpy())])
                    tmp_list.append([node_idx, predict_class])
                # print("tmp list", tmp_list)
            print("acc = ", cnt/baseline.test_index.size(0))

            path = osp.join('./results/target_attack', data_name)
            if not osp.isdir(path):
                os.makedirs(path)
            file = f'{path}/train_percent_{args.train_percent}_corrected_test_ID_res.csv'
            print(" test id list file saved at", file)
            df = pd.DataFrame(tmp_list,columns=cols)
            if not os.path.isfile(file):
                df.to_csv(file, index=False)

            vis_file = f'{path}/train_percent_{args.train_percent}_baseline_A_X_res.pkl'
            print(" baseline visualization file saved at ", vis_file)
            utils.save_to_file(
                [baseline.data.edge_index.to('cpu'), torch.argmax(output.to('cpu'), dim=1), baseline.data.x[:]], vis_file)

            exit(-1)
        # args.node_idx = baseline.test_index[0].item()
        # print(" node idx ", args.node_idx)
        # print(" output shape ", output.shape, output[args.node_idx], type(output[args.node_idx]))
        # pred_class = torch.argmax(output[args.node_idx], dim=0).item()
        # print(" pred class = ", pred_class)
        # origin = baseline.data.y[args.node_idx]
        # print(" origin", origin)
        # utils.save_to_file([baseline.data.edge_index.to('cpu'), torch.argmax(output.to('cpu'), dim=1), baseline.data.x[:]], 'baseline'+str(args.node_idx)+'dataset_preprocess.pkl')
        # exit(-1)


    test_acc, test_loss, macro_precision, macro_recall, macro_f1_score, weighted_precision, weighted_recall, weighted_f1_score = utils.evaluate(
        model, baseline.data, baseline.data.test_mask)
    print(" test acc", test_acc)
    columns = ['seed', 'accuracy', 'macro_precision', 'macro_recall', 'macro_f1_score', 'weighted_precision',
               'weighted_recall', 'weighted_f1_score']

    # path = f'results/baseline'
    path = osp.join('./results/baseline', data_name)
    if not osp.isdir(path):
        os.makedirs(path)
    file = f'{path}/train_percent_{args.train_percent}_res.csv'
    print(" res save as ", file)
    # file = f'{path}/.csv'
    res_list = [args.seed,
                test_acc,
                macro_precision,
                macro_recall,
                macro_f1_score,
                weighted_precision,
                weighted_recall,
                weighted_f1_score]
    df = pd.DataFrame([res_list],columns=columns)
    if not os.path.isfile(file):
        df.to_csv(file, index=False)
    else:
        prev_res = pd.read_csv(file)
        final_res = pd.concat([df, prev_res],ignore_index = True)
        final_res.reset_index()
        final_res.to_csv(file, index=False)
