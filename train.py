import math
import logging
import time
import sys
import random
import argparse
import numpy as np
import torch
import pandas as pd
import math
from graph_test import NeighborFinder
from module import TGAN
from tqdm import tqdm
from utils import EarlyStopMonitor
import os
exp_seed = 0
random.seed(exp_seed)
np.random.seed(exp_seed)
torch.manual_seed(exp_seed)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

### Argument and global variables
parser = argparse.ArgumentParser('Interface for experiments')
parser.add_argument('-d', '--data', type=str, help='data sources to use, try weibo', default='weibo_3600')
parser.add_argument('--bs', type=int, default=30, help='batch_size')
parser.add_argument('--prefix', type=str, default='')
parser.add_argument('--n_degree', type=int, default=20, help='number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2)
parser.add_argument('--n_epoch', type=int, default=3, help='number of epochs')
parser.add_argument('--n_layer', type=int, default=2)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--tune', action='store_true', help='parameters tunning mode, use train-test split on training data only.')
parser.add_argument('--drop_out', type=float, default=0.1, help='dropout probability')
parser.add_argument('--gpu', type=int, default=1, help='idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=None, help='Dimentions of the node embedding')
parser.add_argument('--time_dim', type=int, default=None, help='Dimentions of the time embedding')
parser.add_argument('--feat_dim', type=int, default=64, help='Dimentions of the feature embedding')
parser.add_argument('--agg_method', type=str, choices=['attn', 'lstm', 'mean'], help='local aggregation method', default='attn')
parser.add_argument('--attn_mode', type=str, choices=['prod', 'map'], default='prod')
parser.add_argument('--time', type=str, choices=['time', 'pos', 'empty'], help='how to use time information', default='time')
parser.add_argument('--uniform', action='store_true', help='take uniform sampling from temporal neighbors')

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
UNIFORM = args.uniform
USE_TIME = args.time
AGG_METHOD = args.agg_method
ATTN_MODE = args.attn_mode
SEQ_LEN = NUM_NEIGHBORS
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
FEAT_DIM = args.feat_dim
### set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('log/{}.log'.format(time.ctime()))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)

def eval_one_epoch(num_batch, cas_dict, s_idx, e_idx, tgan, cas, src, dst, ts, label):
    with torch.no_grad():
        tgan = tgan.eval()
        tr_loss = 0
        nb_tr_steps = 0
        preds = []
        labels = []
        for k in tqdm(range(0, num_batch), desc='Inference'):
            cas_l_cut = cas[s_idx:e_idx].to_numpy()
            src_l_cut = src[s_idx:e_idx].to_numpy()
            dst_l_cut = dst[s_idx:e_idx].to_numpy()
            ts_l_cut = ts[s_idx:e_idx].to_numpy()
            label_l_cut = label[s_idx:e_idx].to_numpy()
            #print(train_cas_l[s_idx], cas_l_cut.shape)
            if k < num_batch - 1:
                s_idx = e_idx
                e_idx = s_idx + cas_dict[cas[s_idx]]
            score = tgan.forward(cas_l_cut, src_l_cut, dst_l_cut, ts_l_cut, NUM_NEIGHBORS)
            gt = torch.log2(torch.FloatTensor([label_l_cut[0]]) + 1).to(device)
            loss = criterion(score, gt)

            score = score.detach().cpu().numpy()
            gt = gt.detach().cpu().numpy()

            score = np.squeeze(score).tolist()
            gt = np.squeeze(gt).tolist()

            preds.append(score)
            labels.append(gt)

            tr_loss += loss.item()
            nb_tr_steps += 1

        test_loss = tr_loss / nb_tr_steps
        preds = np.array(preds).reshape(-1, )
        labels = np.array(labels).reshape(-1, )
        MSLE = np.mean(np.square(preds - labels))
        mSLE = np.median(np.square(preds - labels))
    return test_loss, MSLE, mSLE

### Load data and train val test split
g_df = pd.read_csv('./processed/ml_{}.csv'.format(DATA))
g_df.rename(columns={'size':'e_idx'}, inplace = True)
cas_l = g_df.cas
src_l = g_df.src
dst_l = g_df.target
e_l = g_df.e_idx # max=997
ts_l = g_df.ts
label_l = g_df.label

EDGE_NUM = max(e_l) + 1
NODE_NUM = max(max(src_l), max(dst_l)) + 1
cas_dict = dict()

for idx, cas in enumerate(cas_l):
    if cas not in cas_dict.keys():
        cas_dict[cas] = 1
    else:
        cas_dict[cas] += 1
print(len(cas_dict.keys()), min(cas_dict.values()), np.mean(list(cas_dict.values())), max(cas_dict.values()))
cas_num = len(cas_dict.keys())
# spilt train, val, test by cas_id to 7:1.5:1.5
val_cas_split = list(cas_dict.keys())[int(cas_num * 0.7)]
test_cas_split = list(cas_dict.keys())[int(cas_num * 0.85)]
print(val_cas_split, test_cas_split)
valid_train_flag = (cas_l <= val_cas_split) 
valid_test_flag = (cas_l > test_cas_split) 

train_cas_l = cas_l[valid_train_flag]
train_src_l = src_l[valid_train_flag]
train_dst_l = dst_l[valid_train_flag]
train_ts_l = ts_l[valid_train_flag]
train_e_l = e_l[valid_train_flag]
train_label_l = label_l[valid_train_flag]

test_cas_l = cas_l[valid_test_flag].reset_index(drop=True)
test_src_l = src_l[valid_test_flag].reset_index(drop=True)
test_dst_l = dst_l[valid_test_flag].reset_index(drop=True)
test_ts_l = ts_l[valid_test_flag].reset_index(drop=True)
test_e_l = e_l[valid_test_flag].reset_index(drop=True)
test_label_l = label_l[valid_test_flag].reset_index(drop=True)

train_cas_num = len(set(train_cas_l))
test_cas_num = len(set(test_cas_l))
print("train_cas_num: {}, test_cas_num: {}".format(train_cas_num, test_cas_num))

### Initialize the data structure for graph and edge sampling

train_max_idx = max(max(train_src_l), max(train_dst_l))
train_adj_list = [[] for _ in range(train_max_idx + 1)]
for cas, src, dst, eidx, ts in zip(train_cas_l, train_src_l, train_dst_l, train_e_l, train_ts_l):
    train_adj_list[src].append((cas, dst, eidx, ts))
    train_adj_list[dst].append((cas, src, eidx, ts))
train_ngh_finder = NeighborFinder(train_adj_list, uniform=UNIFORM)
print("train ngh_finder finish.")

test_max_idx = max(max(test_src_l), max(test_dst_l))
test_adj_list = [[] for _ in range(test_max_idx + 1)]
for cas, src, dst, eidx, ts in zip(test_cas_l, test_src_l, test_dst_l, test_e_l, test_ts_l):
    test_adj_list[src].append((cas, dst, eidx, ts))
    test_adj_list[dst].append((cas, src, eidx, ts))
test_ngh_finder = NeighborFinder(test_adj_list, uniform=UNIFORM)
print("test ngh_finder finish.")

# a,b,c=train_ngh_finder.get_temporal_neighbor(train_cas_l[:30], train_src_l[:30], train_ts_l[:30], num_neighbors=20)
# print(train_src_l[:30], a)

device = torch.device('cuda:{}'.format(GPU))
tgan = TGAN(train_ngh_finder, node_num=NODE_NUM, edge_num=EDGE_NUM, feat_dim=FEAT_DIM, device=device,
            num_layers=NUM_LAYER, use_time=USE_TIME, agg_method=AGG_METHOD, attn_mode=ATTN_MODE,
            seq_len=SEQ_LEN, n_head=NUM_HEADS, drop_out=DROP_OUT)
optimizer = torch.optim.Adam(tgan.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.MSELoss()
tgan = tgan.to(device)

num_instance = len(train_src_l)
num_batch = train_cas_num
logger.info('num of training instances: {}'.format(num_instance))
logger.info('num of batches per epoch: {}'.format(num_batch))
early_stopper = EarlyStopMonitor()

for epoch in range(NUM_EPOCH):
    # Training 
    # use a cas graph as a batch
    tgan.ngh_finder = train_ngh_finder
    s_idx, e_idx = 0, cas_dict[train_cas_l[0]]
    logger.info('start {} epoch'.format(epoch))
    tr_loss = 0
    nb_tr_steps = 0
    for k in tqdm(range(num_batch), desc='Train'):
        # TODO: 目前选择一个cas的所有参与节点建图，可以修改为只采样重要节点
        cas_l_cut = train_cas_l[s_idx:e_idx].to_numpy()
        src_l_cut = train_src_l[s_idx:e_idx].to_numpy()
        dst_l_cut = train_dst_l[s_idx:e_idx].to_numpy()
        ts_l_cut = train_ts_l[s_idx:e_idx].to_numpy()
        label_l_cut = train_label_l[s_idx:e_idx].to_numpy()
        # print(train_cas_l[s_idx], cas_l_cut.shape)
        if k < num_batch - 1:
            s_idx = e_idx
            e_idx = s_idx + cas_dict[train_cas_l[s_idx]]
        tgan = tgan.train()
        score = tgan.forward(cas_l_cut, src_l_cut, dst_l_cut, ts_l_cut, NUM_NEIGHBORS)
        gt = torch.log2(torch.FloatTensor([label_l_cut[0]]) + 1).to(device)
        loss = criterion(score, gt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tr_loss += loss.item()
        nb_tr_steps += 1
    
    train_loss = tr_loss / nb_tr_steps
    # validation phase
    tgan.ngh_finder = test_ngh_finder
    test_s_idx, test_e_idx = 0, cas_dict[test_cas_l[0]]
    test_loss, MSLE, mSLE = eval_one_epoch(test_cas_num, cas_dict, test_s_idx, test_e_idx, tgan, test_cas_l, test_src_l, test_dst_l, test_ts_l, test_label_l)
    logger.info('end {} epoch, train loss is {}, test loss is {}, MSLE is {}, mSLE is {}'.format(epoch, train_loss, test_loss, MSLE, mSLE))
    