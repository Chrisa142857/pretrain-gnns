import argparse

# from loader import BioDataset
# from dataloader import DataLoaderFinetune
# from splitters import random_split, species_split

import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np
from util import ExtractSubstructureContextPair

from model import GNN, GNN_graphpred
# from sklearn.metrics import roc_auc_score
import graph_prompt as Prompt
# import pandas as pd
# import os
# import pickle
# import wandb

from hcp_data_utils import HCPAScFcDatasetOnDisk
from torch_geometric.loader import DataLoader
from tqdm import trange
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import KFold

# criterion = nn.BCEWithLogitsLoss()
criterion = nn.CrossEntropyLoss()

def train(args, model, device, loader, optimizer, prompt=None):
    model.train()

    # for step, batch in enumerate(tqdm(loader, desc="Iteration")):
    for step, batch in enumerate(loader):
        batch, y = batch['data'].to(device), batch['label'].to(device)
        if prompt is not None:
            pred = model(batch, prompt)
        else:
            pred = model(batch)
        # y = batch.go_target_downstream.view(pred.shape).to(torch.float64)

        optimizer.zero_grad()
        loss = criterion(pred, y)
        loss.backward()

        optimizer.step()


def eval(args, model, device, loader, prompt=None):
    model.eval()
    y_true = []
    y_scores = []

    # for step, batch in enumerate(tqdm(loader, desc="Iteration")):
    for step, batch in enumerate(loader):
        batch, y = batch['data'].to(device), batch['label']

        with torch.no_grad():
            if prompt is not None:
                pred = model(batch, prompt)
            else:
                pred = model(batch)

        y_true.append(y)
        y_scores.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_scores = torch.cat(y_scores, dim = 0).numpy().argmax(1)
    acc = accuracy_score(y_true, y_scores)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_scores, average='weighted')
    return acc, prec, rec, f1
    # roc_list = []
    # for i in range(y_true.shape[1]):
    #     #AUC is only defined when there is at least one positive data.
    #     if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == 0) > 0:
    #         roc_list.append(roc_auc_score(y_true[:,i], y_scores[:,i]))
    #     else:
    #         roc_list.append(np.nan)

    # return np.array(roc_list) #y_true.shape[1]

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=2,
                        help='Which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='Weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='Number of GNN message passing layers (default: 4).')
    parser.add_argument('--emb_dim', type=int, default=116,
                        help='Embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0,
                        help='Dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="max",
                        help='Graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='How the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--model_file', type=str, default = 'neuroscience/model_weights/graphview%s_contextpred.pth.pth', help='File path to read the model (if there is any)')
    parser.add_argument('--gnn_type', type=str, default="GCNConv")
    parser.add_argument('--tuning_type', type=str, default="gpf", help='\'gpf\' for GPF and \'gpf-plus\' for GPF-plus in the paper')
    parser.add_argument('--seed', type=int, default=142857, help = "Seed for splitting dataset.")
    parser.add_argument('--runseed', type=int, default=142857, help = "Seed for running experiments.")
    parser.add_argument('--num_workers', type=int, default = 16, help='Number of workers for dataset loading')
    parser.add_argument('--etrain', type=int, default = 1, help='Evaluating training or not')
    parser.add_argument('--split', type=str, default = "species", help='The way of dataset split(e.g., \'species\' for bio data)')
    parser.add_argument('--num_layers', type=int, default = 1, help='A range of [1,2,3]-layer MLPs with equal width')
    parser.add_argument('--pnum', type=int, default = 5, help='The number of independent basis for GPF-plus')
    parser.add_argument('--max_patience', type=int, default = 30)
    parser.add_argument('--graph_view1', type=str, default = 'SC')
    parser.add_argument('--graph_view2', type=str, default = 'SC')
    parser.add_argument('--use_gpf', action='store_true')
    args = parser.parse_args()
    # args.model_file = args.model_file % args.graph_view
    # args.model_file = 'neuroscience/model_weights/graphviewFC-SC_multicontextpred_onlyrest.pth.pth'
    # args.model_file = 'neuroscience/model_weights/graphviewFC-SC_multicontextpred_onlyrest.pth.pth'
    # args.model_file = 'neuroscience/model_weights/graphviewSC-FC_contextpred_onlyrest.pth.pth'
    args.model_file = ""
    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)


    in_dim = 116
    num_tasks = 4
    dataset = HCPAScFcDatasetOnDisk('AAL_116', adj_type=args.graph_view1, node_attr=args.graph_view2, pretain=False)
    # print(dataset)

    all_subjects = dataset.data_subj

    # Define 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=142857)

    # Initialize lists to store evaluation metrics
    accuracies = []
    f1_scores = []
    prec_scores = []
    rec_scores = []

    for fold, (train_index, index) in enumerate(kf.split(all_subjects)):
        
        train_subjects = [all_subjects[i] for i in train_index]
        subjects = [all_subjects[i] for i in index]

        # Filter dataset based on training and validation subjects
        train_data = [di for di, subj in enumerate(dataset.fc_subject) if subj in train_subjects]
        data = [di for di, subj in enumerate(dataset.fc_subject) if subj in subjects]
        print(f'Fold {fold + 1}, Train {len(train_subjects)} subjects, Val {len(subjects)} subjects, len(train_data)={len(train_data)}, len(data)={len(data)}')
        train_dataset = torch.utils.data.Subset(dataset, train_data)
        valid_dataset = torch.utils.data.Subset(dataset, data)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        loader = DataLoader(valid_dataset, batch_size=10*args.batch_size, shuffle=False, num_workers=args.num_workers)

        #set up model
        model = GNN_graphpred(args.num_layer, in_dim, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type)

        if args.model_file != "":
            model.from_pretrained(args.model_file)
        model.to(device)
        if not args.use_gpf:
            prompt = None
        else:
            if args.tuning_type == 'gpf':
                prompt = Prompt.SimplePrompt(args.emb_dim).to(device)
            elif args.tuning_type == 'gpf-plus':
                prompt = Prompt.GPFplusAtt(args.emb_dim, args.pnum).to(device)

        #set up optimizer
        model_param_group = []
        if prompt is not None:
            model_param_group.append({"params": prompt.parameters(), "lr": args.lr})
        else:
            model_param_group.append({"params": model.gnn.parameters(), "lr": args.lr})
        if args.graph_pooling == "attention":
            model_param_group.append({"params": model.pool.parameters(), "lr": args.lr})
        model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr": args.lr})
        # model_param_group.append({"params": model.encoder.parameters(), "lr": args.lr})
        optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
        # print(optimizer)
        best_f1 = 0
        patience = 0
        for epoch in (pbar := trange(1, args.epochs+1, desc='Epoch')):
            train(args, model, device, train_loader, optimizer, prompt)
            acc, prec, rec, f1 = eval(args, model, device, loader, prompt)
            pbar.set_description(f'Accuracy: {acc}, F1 Score: {f1}, Epoch')
            if f1 >= best_f1:
                if f1 > best_f1: 
                    patience = 0
                else:
                    patience += 1
                best_f1 = f1
                best_acc = acc
                best_prec = prec
                best_rec = rec
            else:
                patience += 1
            if patience > args.max_patience: break
        accuracies.append(best_acc)
        f1_scores.append(best_f1)
        prec_scores.append(best_prec)
        rec_scores.append(best_rec)
        print(f'Accuracy: {best_acc}, F1 Score: {best_f1}, Prec: {best_prec}, Rec: {best_rec}')

    # Calculate mean and standard deviation of evaluation metrics
    mean_accuracy = sum(accuracies) / len(accuracies)
    std_accuracy = torch.std(torch.tensor(accuracies))
    mean_f1_score = sum(f1_scores) / len(f1_scores)
    std_f1_score = torch.std(torch.tensor(f1_scores))
    mean_prec_score = sum(prec_scores) / len(prec_scores)
    std_prec_score = torch.std(torch.tensor(prec_scores))
    mean_rec_score = sum(rec_scores) / len(rec_scores)
    std_rec_score = torch.std(torch.tensor(rec_scores))

    print(f'Mean Accuracy: {mean_accuracy}, Std Accuracy: {std_accuracy}')
    print(f'Mean F1 Score: {mean_f1_score}, Std F1 Score: {std_f1_score}')
    print(f'Mean prec Score: {mean_prec_score}, Std prec Score: {std_prec_score}')
    print(f'Mean rec Score: {mean_rec_score}, Std rec Score: {std_rec_score}')


if __name__ == "__main__":
    main()
