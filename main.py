from os.path import join
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
import nni
from nni.utils import merge_parameter

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree 
from splitter import scaffold_split, random_split, imbalanced_split


from config import args
from datasets.molnet import MoleculeDataset
from pruning.molpeg import MolPeg, cal_scores
from model.gnn import GNN
from model.mlp import MLP
from utils import update_model
import time

def compute_degree(train_dataset):
    # Compute the maximum in-degree in the training data.
    max_degree = -1
    for data in train_dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        max_degree = max(max_degree, int(d.max()))

    # Compute the in-degree histogram tensor
    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    for data in train_dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())
    
    return deg

def seed_all(seed):
    if not seed:
        seed = 0
    print("[ Using Seed : ", seed, " ]")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

def get_num_task(dataset):
    # Get output dimensions of different tasks
    if dataset == 'tox21':
        return 12
    elif dataset in ['hiv', 'bace', 'bbbp']:
        return 1
    elif dataset == 'muv':
        return 17
    elif dataset == 'toxcast':
        return 617
    elif dataset == 'sider':
        return 27
    elif dataset == 'clintox':
        return 2
    elif dataset == 'pcba':
        return 92
    raise ValueError('Invalid dataset name.')

def compute_loss_div(h, h_ref, loss, loss_ref, step):
    sign = 0
    scores = torch.abs(loss -  loss_ref).view(-1)
    return scores, sign
    
# TODO: clean up
def train_general(model, device, loader, optimizer):
    model.train()
    total_loss = 0
    sign_sum = 0

    for step, (inputs, indices, weight) in enumerate(loader):
        for idx in indices:
            idx_stat[idx] += 1
        inputs = inputs.to(device)
        rescale_weight = weight.to(device)
        h = global_mean_pool(model(inputs), inputs.batch)
        pred = output_layer(h)
        
        with torch.no_grad():
            h_ref = global_mean_pool(model_ref(inputs), inputs.batch)
            pred_ref = output_layer(h_ref)
        
        y = inputs.y.view(pred.shape).to(torch.float64)
        is_valid = y ** 2 > 0
        
        loss_mat = criterion(pred.double(), (y + 1) / 2)
        loss_mat = torch.where(
            is_valid, loss_mat,
            torch.zeros(loss_mat.shape).to(device).to(loss_mat.dtype))
        
        # ======== Reference Model =======
        loss_mat_ref = criterion(pred_ref.double(), (y + 1) / 2)
        # loss matrix after removing null target
        loss_mat_ref = torch.where(
            is_valid, loss_mat_ref,
            torch.zeros(loss_mat_ref.shape).to(device).to(loss_mat_ref.dtype))
        # ==================================
        loss_mat = loss_mat if loss_mat.dim() == 1 else loss_mat.mean(1)
        loss_mat_ref = loss_mat_ref if loss_mat_ref.dim() == 1 else loss_mat_ref.mean(1)

        if args.method == 'molpeg':
            scores, sign = compute_loss_div(pred, pred_ref, loss_mat, loss_mat_ref, step)
            train_dataset.__setscore__(indices.detach().cpu().numpy(), scores.detach().cpu().numpy())
            loss_mat = loss_mat*rescale_weight

        elif args.method == 'rankloss':
            scores = loss_mat
            train_dataset.__setscore__(indices.detach().cpu().numpy(), scores.detach().cpu().numpy())
            loss_mat = loss_mat*rescale_weight
        

        loss_mat = loss_mat*rescale_weight
        
        optimizer.zero_grad()
        loss = torch.sum(loss_mat) / torch.sum(is_valid)
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().item()

        # ==== EMA for reference model ====
        update_model(model, model_ref, beta = args.ema_beta)

    global optimal_loss 
    temp_loss = total_loss / len(loader)
    if temp_loss < optimal_loss:
        optimal_loss = temp_loss
        # save_model(save_best=True)

    return total_loss / len(loader), sign_sum


def eval_general(model, device, loader):
    model.eval()
    y_true, y_scores = [], []
    total_loss = 0

    for step, inputs in enumerate(loader):
        inputs = inputs.to(device)
        with torch.no_grad():
            h = global_mean_pool(model(inputs), inputs.batch)
            pred = output_layer(h)
    
        true = inputs.y.view(pred.shape)

        y_true.append(true)
        y_scores.append(pred)


    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()

    if args.dataset == 'pcba':
        ap_list = []

        for i in range(y_true.shape[1]):
            if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
                # ignore nan values
                is_valid = is_valid = y_true[:, i] ** 2 > 0
                ap = average_precision_score(y_true[is_valid, i], y_scores[is_valid, i])

                ap_list.append(ap)

        if len(ap_list) == 0:
            raise RuntimeError('No positively labeled data available. Cannot compute Average Precision.')
        return sum(ap_list) / len(ap_list), total_loss / len(loader), ap_list

    else:
        roc_list = []
        for i in range(y_true.shape[1]):
            # AUC is only defined when there is at least one positive data.
            if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == -1) > 0:
                is_valid = y_true[:, i] ** 2 > 0
                roc_list.append(eval_metric((y_true[is_valid, i] + 1) / 2, y_scores[is_valid, i]))
            else:
                print('{} is invalid'.format(i))

        if len(roc_list) < y_true.shape[1]:
            print(len(roc_list))
            print('Some target is missing!')
            print('Missing ratio: %f' %(1 - float(len(roc_list)) / y_true.shape[1]))

        return sum(roc_list) / len(roc_list), total_loss / len(loader), roc_list

import matplotlib.pyplot as plt
if __name__ == '__main__':
    params = nni.get_next_parameter()
    args = merge_parameter(args, params)
    print('arguments\t', args)
    seed_all(args.runseed)
    device = torch.device(f'cuda:{args.device}') if torch.cuda.is_available() else torch.device('cpu')

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    # Bunch of classification tasks
    num_tasks = get_num_task(args.dataset)
    dataset_folder = '/your_path/molecule_net/'
    dataset = MoleculeDataset(dataset_folder + args.dataset, dataset=args.dataset)
    
    print(dataset)
    print('=============== Statistics ==============')
    print('Avg degree:{}'.format(torch.sum(degree(dataset.data.edge_index[0])).item()/dataset.data.x.shape[0]))
    print('Avg atoms:{}'.format(dataset.data.x.shape[0]/(dataset.data.y.shape[0]/num_tasks)))
    print('Avg bond:{}'.format((dataset.data.edge_index.shape[1]/2)/(dataset.data.y.shape[0]/num_tasks)))

    eval_metric = roc_auc_score

    if args.split == 'scaffold':
        smiles_list = pd.read_csv(dataset_folder + args.dataset + '/processed/smiles.csv',
                                  header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset, (train_smiles, valid_smiles, test_smiles), (_,_,_) = scaffold_split(
            dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, return_smiles=True)
        print('split via scaffold')
    elif args.split == 'random':
        smiles_list = pd.read_csv(dataset_folder + args.dataset + '/processed/smiles.csv',
                                  header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset, (train_smiles, valid_smiles, test_smiles),_ = random_split(
            dataset, null_value=0, frac_train=0.8, frac_valid=0.1,
            frac_test=0.1, seed=args.seed, smiles_list=smiles_list)
        print('randomly split')
    elif args.split == 'imbalanced':
        smiles_list = pd.read_csv(dataset_folder + args.dataset + '/processed/smiles.csv',
                                  header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset, (train_smiles, valid_smiles, test_smiles), (_,_,_) = imbalanced_split(
            dataset, null_value=0, frac_train=0.7, frac_valid=0.15,
            frac_test=0.15, seed=args.seed, smiles_list=smiles_list)
        print('imbalanced split')
        
    else:
        raise ValueError('Invalid split option.')
    print(train_dataset[0])
    print('Training data length: {}'.format(len(train_smiles)))
    idx_stat = [0 for _ in range(len(train_smiles))]
    sign_stat = [0 for _ in range(args.epochs)]

    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)      

    # set up model 
    model_param_group = []
    model_param_ref = []
    model = GNN(num_layer=args.num_layer, emb_dim=args.emb_dim, drop_ratio=args.dropout_ratio).to(device)
    model_ref = GNN(num_layer=args.num_layer, emb_dim=args.emb_dim, drop_ratio=args.dropout_ratio).to(device)
    output_layer = MLP(in_channels=args.emb_dim, hidden_channels=args.emb_dim, 
                        out_channels=num_tasks, num_layers=1, dropout=0).to(device)
    
    if args.pretrain:
        # model_root = 'GraphCL.pth'
        # model = load_model(args.output_model_dir + model_root, model)
        # model_ref = load_model(args.output_model_dir + model_root, model_ref)
        model_root = 'PubChem_Pretrained.pth'
        model.load_state_dict(torch.load(args.output_model_dir + model_root, map_location='cuda:0'))
        
        print('======= Model Loaded :{}======='.format(model_root))
    model_param_group.append({'params': output_layer.parameters(),'lr': args.lr})
    model_param_group.append({'params': model.parameters(), 'lr': args.lr})
    model_param_ref.append({'params': model_ref.parameters(), 'lr': args.lr})
    
    
    print(model)                
    optimizer = optim.Adam(model_param_group, lr=args.lr,
                           weight_decay=args.decay)
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    train_roc_list, val_roc_list, test_roc_list = [], [], []
    train_loss_list, val_loss_list, test_loss_list = [], [], []
    roc_lists = []
    best_val_roc, best_val_idx = -1, 0
    optimal_loss = 1e10
    es = 0

    train_func = train_general
    eval_func = eval_general

    train_dataset = MolPeg(train_dataset, args.ratio if args.ratio else None, 
                              args.epochs, args.delta, scores = None, method=args.method)
    print(train_dataset.method)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              sampler = train_dataset.pruning_sampler())

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        loss_acc, sign_sum = train_func(model, device, train_loader, optimizer)
        sign_stat[epoch-1] = sign_sum
        print('Epoch: {}\nLoss: {}'.format(epoch, loss_acc))

        train_roc = train_loss = 0
        
        val_roc, val_loss, _ = eval_func(model, device, val_loader)
        test_roc, test_loss, roc_list = eval_func(model, device, test_loader)
        nni.report_intermediate_result(test_roc)

        train_roc_list.append(train_roc)
        val_roc_list.append(val_roc)
        test_roc_list.append(test_roc)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        test_loss_list.append(test_loss)
        roc_lists.append(roc_list)
        print('train: {:.6f}\tval: {:.6f}\ttest: {:.6f}'.format(train_roc, val_roc, test_roc))
        print()

        if val_roc > best_val_roc:
            best_val_roc = val_roc
            best_val_idx = epoch - 1
    end_time = time.time()
    print('Total time: {}'.format(end_time - start_time))
    nni.report_final_result(test_roc_list[best_val_idx])
    print('best train: {:.6f}\tval: {:.6f}\ttest: {:.6f}'.format(train_roc_list[best_val_idx], val_roc_list[best_val_idx], test_roc_list[best_val_idx]))
    print('loss train: {:.6f}\tval: {:.6f}\ttest: {:.6f}'.format(train_loss_list[best_val_idx], val_loss_list[best_val_idx], test_loss_list[best_val_idx]))
    print('single tasks roc list:{}'.format(roc_lists[best_val_idx]))
    np.save('20.npy', idx_stat)
