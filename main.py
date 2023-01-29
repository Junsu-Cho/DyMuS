#!/usr/bin/env python37
# -*- coding: utf-8 -*-

import argparse

import os
import time
import random
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, MultiStepLR
from torch.autograd import Variable
from torch.backends import cudnn

import metric
from model import *

import pandas as pd 

from preprocess import *
from tqdm import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def str2bool(v): 
    if isinstance(v, bool): 
        return v 
    if v.lower() in ('yes', 'true', 't', 'y', '1'): 
        return True 
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): 
        return False 
    else: 
        return True


#################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='taobao')
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--embedding_dim', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--l2', type=float, default=0.0)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--patience', type=int, default=1000)
parser.add_argument('--num_interactions', type=int, default=10)
parser.add_argument('--max_purchase_length', type=int, default=10)
parser.add_argument('--max_cart_length', type=int, default=10)
parser.add_argument('--max_fav_length', type=int, default=10)
parser.add_argument('--max_click_length', type=int, default=10)
parser.add_argument('--r', type=int, default=2)
parser.add_argument('--capsule_length', type=int, default=16)
parser.add_argument('--num_beh', type=int, default=4)
parser.add_argument('--num_epoch', type=int, default=500)
parser.add_argument('--num_negs', type=int, default=99)
parser.add_argument('--save', default=False, type=str2bool)
args = parser.parse_args()

dataset = args.dataset
batch_size = args.batch_size
embedding_dim = args.embedding_dim
lr = args.lr
l2 = args.l2
dropout = args.dropout
patience = args.patience
num_interactions = args.num_interactions
max_purchase_length = args.max_purchase_length
max_cart_length = args.max_cart_length
max_fav_length = args.max_fav_length
max_click_length = args.max_click_length
r = args.r
capsule_length = args.capsule_length
num_beh = args.num_beh
num_epoch = args.num_epoch
num_negs = args.num_negs
save = args.save

k = [10, 20]

##################################################################################

print("dataset             : " + str(dataset), flush=True)
print("batch_size          : " + str(batch_size), flush=True)
print("embedding_dim       : " + str(embedding_dim), flush=True)
print("lr                  : " + str(lr), flush=True)
print("l2                  : " + str(l2), flush=True)
print("dropout             : " + str(dropout), flush=True)
print("patience            : " + str(patience), flush=True)
print("num_interactions    : " + str(num_interactions), flush=True)
print("max_purchase_length : " + str(max_purchase_length), flush=True)
print("max_cart_length     : " + str(max_cart_length), flush=True)
print("max_fav_length      : " + str(max_fav_length), flush=True)
print("max_click_length    : " + str(max_click_length), flush=True)
print("r                   : " + str(r), flush=True)
print("capsule_length      : " + str(capsule_length), flush=True)
print("num_beh             : " + str(num_beh), flush=True)
print("num_epoch           : " + str(num_epoch), flush=True)
print("num_negs            : " + str(num_negs), flush=True)
print("save                : " + str(save), flush=True)


def print_result(perf):
    print('Recall@10: ' + str(round(perf[0], 4)) + ', NDCG@10: ' + str(round(perf[1], 4)), flush=True)
    print('Recall@20: ' + str(round(perf[2], 4)) + ', NDCG@20: ' + str(round(perf[2], 4)), flush=True)
    print('', flush=True)

def main():
    print("Loading Dataset...", flush=True)
    
    purchase_file   = "/data/junsu7463/" + dataset + "/data_purchase_500.csv"
    cart_file       = "/data/junsu7463/" + dataset + "/data_cart_500.csv"
    fav_file        = "/data/junsu7463/" + dataset + "/data_fav_500.csv"
    click_file      = "/data/junsu7463/" + dataset + "/data_click_500.csv"

    print("Processing Dataset...", flush=True)
    
    tr_purchase_labels, tr_purchase_data, tr_cart_data, tr_fav_data, tr_click_data, \
    va_purchase_labels, va_purchase_data, va_cart_data, va_fav_data, va_click_data, \
    te_purchase_labels, te_purchase_data, te_cart_data, te_fav_data, te_click_data, \
    num_users, num_items = preprocess(dataset, purchase_file, cart_file, fav_file, click_file, num_interactions=num_interactions, max_purchase_length=max_purchase_length, max_cart_length=max_cart_length, max_fav_length=max_fav_length, max_click_length=max_click_length)

    # del user_interactions

    assert len(tr_purchase_labels) == len(tr_purchase_data)
    assert len(tr_purchase_labels) == len(tr_cart_data)
    assert len(tr_purchase_labels) == len(tr_fav_data) 
    assert len(tr_purchase_labels) == len(tr_click_data)

    assert len(va_purchase_labels) == len(va_purchase_data)
    assert len(va_purchase_labels) == len(va_cart_data)
    assert len(va_purchase_labels) == len(va_fav_data) 
    assert len(va_purchase_labels) == len(va_click_data)

    assert len(te_purchase_labels) == len(te_purchase_data)
    assert len(te_purchase_labels) == len(te_cart_data)
    assert len(te_purchase_labels) == len(te_fav_data) 
    assert len(te_purchase_labels) == len(te_click_data)

    print("# users: " + str(num_users), flush=True)
    print("# items: " + str(num_items), flush=True)

    # model = DyMuS(num_beh, num_items, embedding_dim, max_purchase_length, max_cart_length, max_fav_length, max_click_length, r, capsule_length, dropout).to(device)
    model = DyMuS_plus(num_beh, num_items, embedding_dim, max_purchase_length, max_cart_length, max_fav_length, max_click_length, r, capsule_length, dropout).to(device)

    optimizer = optim.Adam(model.parameters(), lr, weight_decay=l2)
    
    best_perf = [0.0, 0.0, 0.0, 0.0]
    best_perf_i = 0
    

    for epoch in range(1, num_epoch+1):
        print("Epoch: " + str(epoch), flush=True)
        
        # train for one epoch
        train_loss = trainForEpoch(tr_purchase_labels, tr_purchase_data, tr_cart_data, tr_fav_data, tr_click_data, model, optimizer, batch_size, epoch)

        val_purchase_perf = validate(va_purchase_labels, va_purchase_data, va_cart_data, va_fav_data, va_click_data, model, k, batch_size, validation=True)

        print("Validation (purchase)", flush=True)
        print_result(val_purchase_perf)
        
        
        if val_purchase_perf[1] > best_perf[1]:
            if save:
                torch.save(model.state_dict(), 'best_model_' + dataset + '_' + str(r) + '.pt')

            best_perf = list(val_purchase_perf)
            best_perf_i = epoch
            test_purchase_perf = validate(te_purchase_labels, te_purchase_data, te_cart_data, te_fav_data, te_click_data, model, k, batch_size, validation=False)
            print("## Test (purchase)", flush=True)
            print_result(test_purchase_perf)
            

        print("Best_validation (" + str(best_perf_i) + ")", flush=True)
        print_result(best_perf)

        if best_perf_i + patience < epoch:
            exit(1)


def trainForEpoch(purchase_labels, purchase_data, cart_data, fav_data, click_data, model, optimizer, batch_size, epoch):
    model.train()

    ## Data shuffling
    data = list(zip(purchase_labels, purchase_data, cart_data, fav_data, click_data))
    random.shuffle(data)
    purchase_labels, purchase_data, cart_data, fav_data, click_data = zip(*data)

    criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)
    total_loss = 0.0

    start = time.time()
    num_count = 0

    with trange(0, len(purchase_data), batch_size) as t:
        for i in t:
            t.set_description("Training Epoch %d Batch %d" % (epoch, i))
            optimizer.zero_grad()

            batch_purchase_labels, batch_purchase_list, batch_cart_list, batch_fav_list, batch_click_list = purchase_labels[i: i + batch_size], purchase_data[i: i + batch_size], cart_data[i: i + batch_size], fav_data[i: i + batch_size], click_data[i: i + batch_size]
            
            train_purchase = torch.LongTensor(batch_purchase_list).to(device) # b * l
            train_cart = torch.LongTensor(batch_cart_list).to(device) # b * l
            train_fav = torch.LongTensor(batch_fav_list).to(device) # b * l
            train_click = torch.LongTensor(batch_click_list).to(device) # b * l
            train_purchase_labels = torch.LongTensor(batch_purchase_labels).to(device) # b
            
            ## Training model
            # b * i
            output = model(train_purchase, train_cart, train_fav, train_click) 

            loss = criterion(output, train_purchase_labels)
            
            loss.backward()
            optimizer.step() 
            loss_item = loss.item()

            t.set_postfix(loss=loss_item)
                        
            total_loss += loss_item
        
    return total_loss


def validate(purchase_labels, purchase_data, cart_data, fav_data, click_data, model, k, batch_size, validation):
    model.eval()

    perf = [0.0, 0.0, 0.0, 0.0]
    len_val = 0
    
    avg_recall10 = 0
    avg_ndcg10 = 0

    with torch.no_grad():
        with trange(0, len(purchase_data), batch_size) as t:
            for i in t:
                if validation:
                    t.set_description('Validation Batch %d' % i)
                else:
                    t.set_description('Test Batch %d' % i)

                batch_purchase_labels, batch_purchase_list, batch_cart_list, batch_fav_list, batch_click_list = purchase_labels[i: i + batch_size], purchase_data[i: i + batch_size], cart_data[i: i + batch_size], fav_data[i: i + batch_size], click_data[i: i + batch_size]

                valid_purchase = torch.LongTensor(batch_purchase_list).to(device) # b * l
                valid_cart = torch.LongTensor(batch_cart_list).to(device) # b * l
                valid_fav = torch.LongTensor(batch_fav_list).to(device) # b * l
                valid_click = torch.LongTensor(batch_click_list).to(device) # b * l
                
                valid_purchase_labels = torch.LongTensor(batch_purchase_labels).to(device) # b
                    
                ## Model output for evaluation
                # b * i
                output = model(valid_purchase, valid_cart, valid_fav, valid_click)

                ## for all negs
                purchase_logits = F.softmax(output, dim=-1) # b * i

                len_val += output.shape[0]

                ## Metrics
                recall, ndcg = metric.evaluate(purchase_logits, valid_purchase_labels, k=k)
                
                perf[0] += recall[0]
                perf[1] += ndcg[0]
                perf[2] += recall[1]
                perf[3] += ndcg[1]
                                
                avg_recall10 = perf[0] / len_val
                avg_ndcg10 = perf[1] / len_val
                t.set_postfix(Recall_10=avg_recall10, NDCG_10=avg_ndcg10)


    perf = list(map(lambda x: x / len_val, perf))
    
    return perf


if __name__ == '__main__':
    main()
