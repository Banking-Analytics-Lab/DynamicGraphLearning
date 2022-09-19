from copy import deepcopy
import networkx as nx 
from torch.optim import Adam
from cProfile import label
import numpy
from tkinter import HIDDEN
from sklearn.metrics import classification_report
import time
import os
import pickle
from curses import window
from sklearn.metrics import precision_recall_curve
import optuna
from email.policy import default
from operator import concat
from pyexpat import features
import pandas as pd
import numpy as np
from urllib.parse import MAX_CACHE_SIZE
import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.nn import Linear, LSTM, BCEWithLogitsLoss, GRUCell ,LSTMCell
from torch.optim import SGD
import argparse
from sklearn.preprocessing import StandardScaler
from numpy import genfromtxt
from statistics import mean
import torch.nn.functional as F
from torch.nn import Linear, Dropout
from torch_geometric.nn import GCNConv, GATv2Conv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import auc, roc_curve, classification_report, precision_recall_curve, accuracy_score
import math 
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import random
import pandas as pd
from scipy.spatial.distance import pdist,squareform
from IPython.display import display
def create_weighted_binary_crossentropy(zero_weight, one_weight):

    def weighted_binary_crossentropy(y_true, y_pred):

        # Original binary crossentropy (see losses.py):
        # K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

        # Calculate the binary crossentropy

        bce =  BCEWithLogitsLoss()
        b_ce = bce(y_true, y_pred)

        # Apply the weights
        weight_vector = y_true * one_weight + (1. - y_true) * zero_weight
        weighted_b_ce = weight_vector * b_ce

        # Return the mean error
        return torch.mean(weighted_b_ce)

    return weighted_binary_crossentropy

def recon_upsample(embed, labels,edges,sample_rate,end_index = None):

    n_nodes = len(set(edges.flatten()) )
    avg_number = n_nodes * sample_rate
    chosen = np.argwhere(labels.flatten() == 1).flatten()
    original_idx = embed.shape[0]



    #ipdb.set_trace()   

    c_portion = int(avg_number/chosen.shape[0])

    for j in range(c_portion):
        chosen_embed = embed[chosen,:]
        distance = squareform(pdist(chosen_embed.cpu().detach()))
        np.fill_diagonal(distance,distance.max()+100)
        idx_neighbor = distance.argmin(axis=-1)
        interp_place = random.random()

        new_embed = embed[chosen,:] + (chosen_embed[idx_neighbor,:]-embed[chosen,:])*interp_place
        new_labels = torch.zeros((chosen.shape[0],1)).reshape(-1).fill_(1)
        embed = torch.cat((embed,new_embed), 0)
        labels = torch.cat((torch.tensor(labels),new_labels), 0)

    if end_index is not None: 
        embed = embed[:end_index]
    synthetic_index = [x for x in range( original_idx , embed.shape[0]  ,1) ]

    return embed, labels,synthetic_index


def get_window(dates):
    return [[int(dates[i-1]),int(dates[i]),int(dates[i+1])] for i in range( 1,len(dates) -1 )]

def check_nans(ten): 
    cond = np.isnan(ten).any()
    mask =np.argwhere(np.isnan(ten))
    print('HI?')
    if cond: 
        print( 'THERE ARE NANS')
        print('at indexes' , mask)
        print(ten[mask ])
    else: 
        print('PASSED')
    exit(0)

def rnn_batchless(model_RNN,h,hidd):
    # for tens in h: 
    # hidd = 
    print(h.shape, hidd[0].shape,hidd[1].shape,'SHAPES')
    out_hidd = torch.ones_like(h)
    out_cell = torch.ones_like(h)
    num_padding = h.shape[0]  - hidd[0].shape[0]
    padd_hidd = torch.zeros((num_padding,h.shape[1]))
    padd_cell = torch.zeros((num_padding,h.shape[1]))
    # out_hidd = torch.cat([out_hidd,padd_hidd])
    # out_cell = torch.cat([out_hidd,padd_cell])
    print(out_hidd.shape, out_cell.shape,'SHAPES NEW')
   
    for node in range(hidd[0].shape[0]):
        # print(hidd[1][r,:].shape,hidd[0][r,:].shape, 'FIRST')
        out_hidd[node,:],out_cell[node,:] = model_RNN(h[node,:],(hidd[0][node ,:],hidd[1][node,:]))
        # out1,out2 = model_RNN(h[r,:],(hidd[0][r,:],hidd[1][r,:]))
        # print(out1.shape,out2.shape ,'out',out_hidd.shape,out_cell.shape,'AFTER')
        # exit(0)

    
    # print(out_hidd.shape,out_cell.shape)

    return (out_hidd,out_cell)

def train_foward_only_RNN(model_GAT,model_RNN,model_FFNN,h0,h_dict,window,data_dict,training = None,sample_rate = None): 

        hidden_states = h0 

        for i,month in enumerate(window):
            data = data_dict[month]
            labs = data.y
            h = data.x
            
            if i == 0:
                if isinstance(model_RNN,LSTMClassification):
                    if args.full_batch: 
                        # print('IN line 134')
                        (hidden_states) = rnn_batchless(model_RNN,h,h0)
                    else:
                        print(h.shape, h0[0].shape)

                        (hidden_states) = model_RNN(torch.Tensor(h), (h0[0],h0[1])) 
                    h0 = (hidden_states[0].clone().detach(),hidden_states[1].clone().detach())  
                else: 
                    if args.full_batch: 
                        (hidden_states) = rnn_batchless(model_RNN,h,h0)
                    else:
                        (hidden_states) = model_RNN(torch.Tensor(h), h0)
                    (h0) = hidden_states.clone().detach()
            else:
                if isinstance(model_RNN,LSTMClassification):
                    if args.full_batch: 
                        (hidden_states) = rnn_batchless(model_RNN,h,h0)
                    else:
                        (hidden_states) = model_RNN(torch.Tensor(h), (hidden_states[0],hidden_states[1]))    
                else:
                    if args.full_batch: 
                        (hidden_states) = rnn_batchless(model_RNN,h,h0)
                    else:
                        (hidden_states) = model_RNN(torch.Tensor(h), hidden_states)

        if isinstance(model_RNN,LSTMClassification):    
            if training == 'RNN': 
                hidden_states = hidden_states[0]
                hidden_states,labs,synth_index = recon_upsample(hidden_states,data.y,data.edge_index,sample_rate)
                scores = model_FFNN(hidden_states)
            else:
                scores = model_FFNN(hidden_states[0])
        else: 
            if training == 'RNN' :

                hidden_states,labs,synth_index = recon_upsample(hidden_states,data.y,data.edge_index,sample_rate)
  
            scores = model_FFNN(hidden_states)
        if training:     
            return scores,torch.Tensor(labs),h0,synth_index
        else:
            return scores,torch.Tensor(labs),h0


def train_foward_only_GNN(model_GAT,model_RNN,model_FFNN,h0,h_dict,window,data_dict,training = None,sample_rate = None): 
        
        for i,month in enumerate(window):
            data = data_dict[month]
            labs = data.y
            if isinstance(model_GAT,GAT):
                h = model_GAT(torch.Tensor(data.x).float(), torch.Tensor(data.edge_index).int(),torch.tensor(data.edge_attr).float()) 
            else :
                h = model_GAT(torch.Tensor(data.x).float(), torch.Tensor(data.edge_index).int(),torch.tensor(data.edge_attr).int())
  
            hidden_states = h
            if training == 'RNN': 

                hidden_states,labs,synth_index = recon_upsample(hidden_states,data.y,data.edge_index,sample_rate)
                scores = model_FFNN(hidden_states)
            else:
                scores = model_FFNN(hidden_states)
        
            scores = model_FFNN(hidden_states)

        if training:     
            return scores,torch.Tensor(labs),h0,synth_index
        else:
            return scores,torch.Tensor(labs),h0


def get_batches( windows,data_dict,batch_size): 
    return [[DataLoader(data_dict[i],batch_size= batch_size) for i in x ] for x in windows]

def train(data_dict,ts_list,model_GAT, model_RNN, model_FFNN,n_epoch,lr,\
    momentum,sample_rate,print_epochs = 10,loss = BCEWithLogitsLoss() ,\
    val_h0_test = False, upsample = False,ONLY_RNN = False,train_forward_funcion = None,no_windows = False ,boot_sample = 10 ,early_stop = 10):

    loss_function = loss
    optimizer_decoder = Adam(model_FFNN.parameters(), lr=lr)
    optimizer_encoder = Adam(model_FFNN.parameters(), lr=lr)
    optimizer_FNN = Adam(model_FFNN.parameters(), lr=lr)
    history = {
        'loss': []
    }
    h_zero = model_RNN._init_weights()
 
    if no_windows: 
        windows=    [ts_list[:-3]]
        val_window = [ts_list[-3:-2],ts_list[-2:-1]]
        test_window = [ts_list[-1:]]
        print(windows,val_window,test_window)
    else:
        full_windows = get_window(ts_list)
        windows = full_windows[:-4]
        val_window = full_windows[-4:-2]
        test_window = full_windows[-2:]
        print(windows,val_window,test_window)
    print(windows,val_window,test_window)
    
    h_dict = {}
    test_auc_seen = []
    test_auc_unseen = []
    # test = set(list(data_dict[val_window[-1][-2]].edge_index.flatten())).difference(set(list(data_dict[val_window[-1][-1]].edge_index.flatten())))
 
    train_nodes= set(list(data_dict[windows[-1][-1]].edge_index.flatten()))
    val_nodes = set(list(data_dict[val_window[-1][-1]].edge_index.flatten())).difference(train_nodes)
    test_nodes = set(list(data_dict[test_window[-1][-1]].edge_index.flatten())).difference(train_nodes)

    auc_list = []
    auprc_list = []
    val_auc_seen = []
    val_auc_unseen = []
    val_auprc_seen = []
    val_auprc_unseen = []
    min_val_loss = np.inf
    get_stats = lambda arr:  [np.mean(arr),np.std(arr),np.quantile(arr,0),np.quantile(arr,.25),np.quantile(arr,.5),np.quantile(arr,.75),np.quantile(arr,1)]
    since_changed = 0
    for epoch in tqdm(range(n_epoch)):

        losses = [] 
        loss = 0
        h0 = h_zero # the size of the hidden state for the RNN
        model_GAT.train()
        model_RNN.train()
        model_FFNN.train()

        for  month_window in windows:
            
            optimizer_decoder.zero_grad()
            optimizer_encoder.zero_grad()
            optimizer_FNN.zero_grad()
            if upsample:
                scores,labels,h0,synth_index =\
                    train_forward_funcion(model_GAT,model_RNN,model_FFNN,h0,h_dict,\
                        window = month_window,data_dict=data_dict,sample_rate=sample_rate,training = 'RNN')
            else: 
                scores,labels,h0 =\
                    train_forward_funcion(model_GAT,model_RNN,model_FFNN,h0,h_dict,\
                        window = month_window,data_dict=data_dict,sample_rate=sample_rate)
                synth_index = []

            # scores,labels,h0 =\
            #      train_foward(model_GAT,model_RNN,model_FFNN,h0,h_dict,\
            #          window = month_window,data_dict=data_dict,sample_rate=sample_rate)
            nodes_to_backprop = list(set(data_dict[month_window[-1]].edge_index.flatten())) + synth_index
            # nodes_to_backprop = list(set(data_dict[month_window[-1]].edge_index.flatten())) 


            scores_for_loss = torch.Tensor(scores[nodes_to_backprop]).float().flatten()
            loss = loss_function(scores_for_loss.flatten(), labels[nodes_to_backprop].flatten())
            loss.backward()
            if epoch == 0:
                print('SYNTHETIC NODE STATISTICS')
                print(f'Produced {len(synth_index)} nodes originially had {len(list(set(data_dict[month_window[-1]].edge_index.flatten())))} nodes \n Now have {len(nodes_to_backprop)} total nodes at ts = {month_window[-1]}')
            assert all(labels[synth_index] == 1 )
            optimizer_encoder.step()
            optimizer_decoder.step()
            optimizer_FNN.step()
            losses.append(float(loss))
            avg_loss = np.mean(losses)
            history['loss'].append(avg_loss)
            fpr, tpr, thresholds = roc_curve(labels[nodes_to_backprop], scores_for_loss.detach().numpy().flatten())
            precision, recall, thresholds = precision_recall_curve(labels[nodes_to_backprop], scores_for_loss.detach().numpy().flatten())

            auc_list.append(auc(fpr, tpr))

            auprc_list.append(auc( recall,precision))
            # print(f'INJECTED {len(synth_index)} NODES FOR A TOTAL OF {len( labels[nodes_to_backprop].flatten())} NODES ')
        if epoch % 1 == 0 :
            print(f'Current loss {loss} ')
            print(f'Averages over last {print_epochs} epochs')
            print(f'TRAIN AUC at epoch {epoch}: ')
            auc_mean = mean(auc_list)
            print(auc_mean)
            print(f'TRAIN AUPRC at epoch {epoch}: ')
            auprc_mean = mean(auprc_list)
            print(auprc_mean)

            mask = pd.Series(scores_for_loss.detach().numpy().flatten()) > 0.5
            preds = mask.map(lambda x: 1.0 if x == True else 0.0)
            print(classification_report(labels[nodes_to_backprop], preds, target_names = ['0','1'], output_dict = True))


            model_GAT.eval()
            model_RNN.eval()
            model_FFNN.eval()
            val_loss = 0
            for month_window in val_window:
                nodes_to_eval= list(set(data_dict[month_window[-1]].edge_index.flatten()))
                cur_val_nodes = list(set(nodes_to_eval ) & val_nodes)
                seen_val_nodes = list(set(nodes_to_eval) & train_nodes)

                val_scores,val_labels,val_h0 =\
                    train_forward_funcion(model_GAT,model_RNN,model_FFNN,h0,h_dict,window = month_window,data_dict=data_dict)
                if val_h0_test:
                    h0 = val_h0
                val_scores = val_scores.detach()
                val_loss += loss_function(val_scores.flatten(), val_labels.flatten())
                print(f'VAL LOSS {loss}')
                val_scores_train = val_scores[cur_val_nodes]
                val_labels_seen = val_labels[cur_val_nodes]
                
                val_scores_unseen = val_scores[seen_val_nodes]
                val_labels_unseen = val_labels[seen_val_nodes]


                print(f'{len(seen_val_nodes)=},{len(nodes_to_eval)=},{len(cur_val_nodes)=} , {len(val_nodes)=}, {len(val_scores)=} , {len(val_labels)=}')
                fpr, tpr, thresholds = roc_curve( val_labels_seen,val_scores_train)
                fpr_useen, tpr_unseen, thresholds = roc_curve( val_labels_unseen,val_scores_unseen)

                precision_seen, recall_seen, thresholds = precision_recall_curve(val_labels_seen,val_scores_train)
                precision_unseen, recall_unseen, thresholds = precision_recall_curve(val_labels_unseen,val_scores_unseen)

                val_auc_seen.append(auc(fpr, tpr))
                val_auc_unseen.append(auc(fpr_useen, tpr_unseen))

                val_auprc_seen.append(auc(recall_seen,precision_seen))
                val_auprc_unseen.append(auc(recall_unseen,precision_unseen))

            val_loss = val_loss/len(val_window)
            if min_val_loss > val_loss:
                min_val_loss = val_loss
                best_ffn = deepcopy(model_FFNN.state_dict())
                best_rnn = deepcopy(model_RNN.state_dict())
                best_gnn = deepcopy(model_GAT.state_dict())
                best_epoch = epoch
            
            print(f'VAL LOSS at epoch{epoch}')
            print(val_loss)

            print(f' VAL AUC SEEN at epoch {epoch}:')
            out_val_auc_seen = mean(val_auc_seen)
            print(mean(val_auc_seen))

            print(f'VAL AUC UNSEEN at epoch {epoch}:')
            out_val_auc_unseen = mean(val_auc_unseen)
            print(mean(val_auc_unseen))

            print(f'VAL AUPRC SEEN at epoch {epoch}:')
            print(mean(val_auprc_seen))
            out_val_auprc_seen = mean(val_auprc_seen)

            print(f'VAL AUPRC UNSEEN at epoch {epoch}:')
            print(mean(val_auprc_unseen))
            out_val_auprc_unseen = mean(val_auprc_unseen)

            fpr_all, tpr_all, thresholds = roc_curve(val_labels,val_scores)
            optimal_idx = np.argmax(tpr_all - fpr_all)
            optimal_threshold = thresholds[optimal_idx]
            mask = pd.Series(val_scores.detach().numpy().flatten()) > optimal_threshold
            val_preds = mask.map(lambda x: 1.0 if x == True else 0.0)
            print(classification_report(val_labels, val_preds, target_names = ['0','1'], output_dict = True))

            auc_list = []
            auprc_list = []
            val_auc_seen = []
            val_auc_unseen = []
            val_auprc_seen = []
            val_auprc_unseen = []
            print(since_changed,'since changed')
            if since_changed == early_stop:
                print(f'EARLY STOP AT EPOCH {epoch}')
           
                break
       
            since_changed+=1
    
    model_FFNN.load_state_dict(best_ffn)
    model_GAT.load_state_dict(best_gnn)
    model_RNN.load_state_dict(best_rnn)
    val_loss = 0

    val_scores_seen_list = []
    val_scores_unseen_list = []

    val_labels_seen_list = []
    val_labels_unseen_list = []
    for month_window in val_window:
        nodes_to_eval= list(set(data_dict[month_window[-1]].edge_index.flatten()))
        cur_val_nodes = list(set(nodes_to_eval ) & val_nodes)
        seen_val_nodes = list(set(nodes_to_eval) & train_nodes)

        val_scores,val_labels,val_h0 =\
            train_forward_funcion(model_GAT,model_RNN,model_FFNN,h0,h_dict,window = month_window,data_dict=data_dict)
        if val_h0_test:
            h0 = val_h0
        val_scores = val_scores.detach()
        val_loss += loss_function(val_scores.flatten(), val_labels.flatten())
        print(f'VAL LOSS {loss}')
        val_scores_train = val_scores[cur_val_nodes]
        val_labels_seen = val_labels[cur_val_nodes]
        
        val_scores_unseen = val_scores[seen_val_nodes]
        val_labels_unseen = val_labels[seen_val_nodes]

        val_scores_seen_list.append(val_scores_train)
        val_scores_unseen_list.append(val_scores_unseen)

        val_labels_seen_list.append(val_labels_seen)
        val_labels_unseen_list.append(val_labels_unseen)

        print(f'{len(seen_val_nodes)=},{len(nodes_to_eval)=},{len(cur_val_nodes)=} , {len(val_nodes)=}, {len(val_scores)=} , {len(val_labels)=}')
        fpr, tpr, thresholds = roc_curve( val_labels_seen,val_scores_train)
        fpr_useen, tpr_unseen, thresholds = roc_curve( val_labels_unseen,val_scores_unseen)

        precision_seen, recall_seen, thresholds = precision_recall_curve(val_labels_seen,val_scores_train)
        precision_unseen, recall_unseen, thresholds = precision_recall_curve(val_labels_unseen,val_scores_unseen)

        val_auc_seen.append(auc(fpr, tpr))
        val_auc_unseen.append(auc(fpr_useen, tpr_unseen))

        val_auprc_seen.append(auc(recall_seen,precision_seen))
        val_auprc_unseen.append(auc(recall_unseen,precision_unseen))
    
    
    val_scores_train_boot = np.concatenate([np.array(x) for x in val_scores_seen_list])
    val_scores_unseen_boot = np.concatenate([np.array(x) for x in val_scores_unseen_list])

    val_labels_seen_boot = np.concatenate([np.array(x) for x in val_labels_seen_list])
    val_labels_unseen_boot = np.concatenate([np.array(x) for x in val_labels_unseen_list])

    boots_seen = bootstrap_preds(val_scores_train_boot,val_labels_seen_boot,boot_sample)
    boots_unseen = bootstrap_preds(val_scores_unseen_boot,val_labels_unseen_boot,boot_sample)
    val_auc_seen_stats = get_stats(boots_seen)
    val_auc_unseen_stats = get_stats(boots_unseen)




    val_loss = val_loss/len(val_window)

    print(f'BEST MODEL SEEN AT {best_epoch}')

    print(f'VAL LOSS at epoch {best_epoch}')
    print(val_loss)

    print(f' VAL AUC SEEN at epoch {best_epoch}:')
    out_val_auc_seen = mean(val_auc_seen)
    print(mean(val_auc_seen))

    print(f'VAL AUC UNSEEN at epoch {best_epoch}:')
    out_val_auc_unseen = mean(val_auc_unseen)
    print(mean(val_auc_unseen))

    print(f'VAL AUPRC SEEN at epoch {best_epoch}:')
    print(mean(val_auprc_seen))
    out_val_auprc_seen = mean(val_auprc_seen)

    print(f'VAL AUPRC UNSEEN at epoch {best_epoch}:')
    print(mean(val_auprc_unseen))
    for month_window in test_window:
            nodes_to_eval= list(set(data_dict[month_window[-1]].edge_index.flatten()))
            cur_test_nodes = list(set(nodes_to_eval ) & test_nodes)
            seen_test_nodes = list(set(nodes_to_eval) & train_nodes)

            test_scores,test_labels,_ =\
             train_forward_funcion(model_GAT,model_RNN,model_FFNN,h0,h_dict,window = month_window,data_dict=data_dict)

            test_scores = test_scores.detach()
                
            test_scores_train = test_scores[cur_test_nodes]
            test_labels_seen = test_labels[cur_test_nodes]
                
            test_scores_unseen = test_scores[seen_test_nodes]
            test_labels_unseen = test_labels[seen_test_nodes]

            fpr, tpr, thresholds = \
                roc_curve(test_labels_seen,test_scores_train)
            fpr_unseen, tpr_unseen, thresholds =\
                 roc_curve(test_labels_unseen,test_scores_unseen)
            
            fpr_all, tpr_all, thresholds = roc_curve(test_labels,test_scores)
            optimal_idx = np.argmax(tpr_all - fpr_all)

            optimal_threshold = thresholds[optimal_idx]

            mask = pd.Series(val_scores.detach().numpy().flatten()) > optimal_threshold
            val_preds = mask.map(lambda x: 1.0 if x == True else 0.0)

            print(classification_report(val_labels, val_preds, target_names = ['0','1'], output_dict = True))
            precision_seen, recall_seen, thresholds = precision_recall_curve(test_labels_seen,test_scores_train)
            precision_unseen, recall_unseen, thresholds = precision_recall_curve(test_labels_unseen,test_scores_unseen)

            print(f'TEST AUC UNSEEN at epoch {epoch}:')
            test_auc_seen.append(auc(fpr, tpr))
            test_auc_unseen.append(auc(fpr_unseen, tpr_unseen))


    boots_seen_test = bootstrap_preds(test_scores_train,test_labels_seen,boot_sample)
    boots_unseen_test = bootstrap_preds(test_scores_unseen,test_labels_unseen,boot_sample)
    test_auc_seen_stats = get_stats(boots_seen_test)
    test_auc_unseen_stats = get_stats(boots_unseen_test)

    print('TEST AUC SEEN:')
    print(mean(test_auc_seen))
    print(f'TEST AUC UNSEEN :')
    print(mean(test_auc_unseen))
    print('TEST AUPRC SEEN:')
    test_auprc_seen =auc(  recall_seen,precision_seen)
    print(test_auprc_seen)
    test_auprc_unseen = auc(recall_unseen,precision_unseen)
    print('TEST AUPRC UNSEEN:')
    print(test_auprc_unseen)
    mask = pd.Series(test_scores.detach().numpy().flatten()) > 0.5
    test_preds = mask.map(lambda x: 1.0 if x == True else 0.0)
    print(classification_report(test_labels, test_preds, target_names = ['0','1'], output_dict = True))
    out_l = flatten_list([out_val_auc_seen,out_val_auc_unseen,out_val_auprc_seen,out_val_auprc_unseen,val_auc_seen_stats,val_auc_unseen_stats,\
        mean(test_auc_seen),mean(test_auc_unseen),test_auprc_seen,test_auprc_unseen,test_auc_seen_stats,test_auc_unseen_stats,auc_mean,auprc_mean])


    return [model_FFNN,model_GAT,model_RNN],out_l


def flatten_list(_2d_list):
    flat_list = []
    # Iterate through the outer list
    for element in _2d_list:
        if type(element) is list:
            # If the element is of type list, iterate through the sublist
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list

def bootstrap_preds(preds, labs,num_boot = 10000):

    n = len(preds)
    boot_means = np.zeros(num_boot)
    data = pd.DataFrame({'preds':preds.flatten(),'labs':labs.flatten()})

    np.random.seed(0)
    for i in range(num_boot):
        d = data.sample(n, replace=True)

        fpr, tpr, thresholds = roc_curve(d.labs,d.preds)
        
        boot_means[i] = auc(fpr,tpr)

    return boot_means


def get_data_node_classification(dataset_name, GCN,full_batch,base,page_rank = False):
### Load data and train val test split

    graph_df = pd.read_csv(base + 'data/ml_{}.csv'.format(dataset_name))
    edge_features = np.load(base + 'data/ml_{}.npy'.format(dataset_name))
    inds = np.argwhere(np.sum(edge_features,axis = 1) ==0 ).flatten()
    #equals 1 due to added edge to match node indexing
    assert len(inds.flatten()) == 1, 'EDGES FROM REFERALS IN DATASET'

    node_features = np.load(base + 'data/ml_{}_node.npy'.format(dataset_name))


    n_nodes = node_features.shape[0]
    label_arr = np.zeros((n_nodes,1,len(graph_df.ts.unique())))
    labels = graph_df.label.values
    datas = {}
    ts_iter = sorted(graph_df.ts.unique())[4:]
    sum_rows = 0 

    assert not all([np.isnan(x).any() for x in [node_features,edge_features,graph_df.to_numpy()]])

    for ts in ts_iter:
        sub_graph = graph_df[graph_df.ts == ts]
        sources = sub_graph.u.values
        destinations = sub_graph.i.values
        edges = np.array([sources,destinations])
        labels = make_labels(sub_graph,label_arr,ts)
        nodes = list(set(edges.flatten()) ) 
        if page_rank:
            n_feats = make_page_rank(edges,node_features[:,:,int(ts)])
        else:
            n_feats = node_features[:,:,int(ts)]
       
   
        print(len(labels),'labels')
        assert (edges.shape[1] == len(sub_graph.index)), 'MISMATCH IN EDGE FEATS AND EDGE SIZE '
        if GCN == 'GAT': 
            datas[ts]  = Data(n_feats,edges,y = labels[:,:,int(ts)].flatten(),edge_attr = edge_features[sub_graph.index,:],nodes = nodes )
        elif GCN == 'GCN': 
            datas[ts]  = Data(n_feats ,edges,y = labels[:,:,int(ts)].flatten(),edge_attr = np.sum(edge_features[sub_graph.index,:],axis = 1),nodes = nodes)
        sum_rows+= edges.shape[1]
        print(f'THE MONTH {ts} has {len(edges.flatten())} edges \n it has {len(set(edges.flatten()))} nodes with {len(np.argwhere(labels[:,:,int(ts)].flatten() == 1).flatten())} postives')
        print(f'NODE FEATRUE MATRIX IS DIMENSIONS {n_feats.shape}')
    assert sum_rows == len(graph_df[graph_df.ts.isin(ts_iter)]), 'NOT SAME NODES AS IN EDGES'

    return datas,ts_iter

def make_page_rank(edges,data):
    edge_df = pd.DataFrame({'u': edges[0,:], 'v': edges[1,:]})

    data_graph=nx.from_pandas_edgelist(edge_df,source = 'u',target = 'v', create_using= nx.Graph)

    h = pd.DataFrame(data)
    h['PageRank'] = 0
    total_nodes = data.shape[0]
    #print(total_nodes)

    for i in nx.connected_components(data_graph):
        subgraph = data_graph.subgraph(list(i))
        #print(subgraph.number_of_nodes())
        if subgraph.number_of_nodes() != 1:
            pagerank_dict = nx.pagerank(subgraph)
            for k in subgraph:
                h['PageRank'].loc[k] = pagerank_dict[k]*subgraph.number_of_nodes()/total_nodes
    return h.to_numpy()

def make_labels(df,label_arr,ts): 
    mask = df[df.label == 1].u.values
    # if ts == 4:
    #     print(df)
    #     print(len(set(mask)),len(df[df.label != 1].u.values),len(df),'\n',mask)
    
        
    label_arr[mask,0,int(ts)] = 1 

    return label_arr
  

class GCN(torch.nn.Module):
  """Graph Convolutional Network"""
  def __init__(self, dim_in, dim_h, dim_out,num_layers):
    super().__init__()
    self.gcn1 = GCNConv(dim_in, dim_h)
    self.GCN_list = torch.nn.ModuleList([GCNConv(dim_h, dim_h)  for _ in range(num_layers)])
    self.lin = Linear(dim_h, dim_out)


  def forward(self, x, edge_index,edge_feats):
    #h = F.dropout(x, p=0.5, training=self.training)
    h = self.gcn1(x, edge_index,edge_weight = edge_feats)
    h = F.elu(h)
    h = F.dropout(h, .2)
    for l in self.GCN_list:
        h = l(h, edge_index,edge_weight = edge_feats)
        h = F.elu(h)
        h = F.dropout(h, .2)
    h = self.lin(h)
    return h

class GAT(torch.nn.Module):
    def __init__(self, dim_in, dim_h, dim_out, edge_dim,heads=8,num_layers = 5 ):
        super().__init__()
        self.gat1 = GATv2Conv(dim_in, dim_h, heads=heads,edge_dim = edge_dim) # dim_h * num heads
        #dim_h * heads > dim_h
        self.GAT_list = torch.nn.ModuleList([GATv2Conv(dim_h*heads, dim_h, heads=heads,edge_dim = edge_dim)  for _ in range(num_layers)])
        self.lin = Linear(dim_h*heads, dim_out)

    def forward(self, x, edge_index,edge_feats):

        h = self.gat1(x, edge_index,edge_attr = edge_feats)
        h = F.elu(h)
        h = F.dropout(h, .2)
        for l in self.GAT_list:
            h = l(h, edge_index,edge_attr = edge_feats)
            h = F.elu(h)
            h = F.dropout(h, .2)
        h = self.lin(h)
        return h

class LSTMClassification(torch.nn.Module):

    def __init__(self,  input_dim,hidden_dim,n_nodes):
        super(LSTMClassification, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.n_nodes = n_nodes
        self.lstm = LSTMCell( input_dim,hidden_dim)


    def forward(self, input_, h0):
        (h, c) = self.lstm(input_, h0)
        return (h,c)
    def _init_weights( self):
        h0 = torch.ones(self.n_nodes ,self.hidden_dim)
        c0 = torch.ones(self.n_nodes, self.hidden_dim)

        
        return (h0,c0)

class GRUClassification(torch.nn.Module):

    def __init__(self, input_dim,hidden_dim,n_nodes):
        super(GRUClassification, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_nodes = n_nodes
        self.input_dim = input_dim
        self.gru = GRUCell(input_size =  input_dim,hidden_size = hidden_dim)

    def forward(self, input_, h0):

        h = self.gru(input_, h0)
        
        return h
    def _init_weights( self):
        h0 = torch.randn(self.n_nodes,self.hidden_dim)

        return h0

class FFNN(torch.nn.Module):

    def __init__(self, hidden_dim, target_size):
        super(FFNN, self).__init__()

        self.fc = Linear(hidden_dim, int(hidden_dim/2))
        
        self.fc2 = Linear(int(hidden_dim/2), target_size)

    def forward(self, input_):
        h = self.fc(input_)
        h = F.relu(h)
        h = self.fc2(h)

        return h


parser = argparse.ArgumentParser(description='GNN parser')
parser.add_argument('--GNN', default = 'GCN',
                    help='what GNN architechture to use')
parser.add_argument('--RNN', default = 'LSTM',
                    help='what RNN architechture to use')
parser.add_argument('--hidden_dim', default = 2,
                    help='Hidden dim for RNN', type = int)
parser.add_argument('--embedding_dim', default = 2,
                    help='embedding out dim for GNN', type = int)
parser.add_argument('--GNN_layers', default = 2,
                    help='how many conv layers to use, will be +1 as base layer is not included',type = int)
parser.add_argument('--epochs', default =3,
                    help='epochs',type = int)
parser.add_argument('--run',
                    help='epochs',type = str)
parser.add_argument('-full_batch', action='store_true',
                    help='to run LSTM /GRU on batch or for loop basis')
parser.add_argument('-no_windows', action='store_true',
                    help='to run LSTM /GRU on batch or for loop basis')
parser.add_argument('-val_h0_test', action='store_true',
                    help='to run LSTM /GRU on batch or for loop basis')
parser.add_argument('-upsample', action='store_true',
                    help='to run LSTM /GRU on batch or for loop basis')                    

parser.add_argument('-ONLY_RNN', action='store_true',
                    help='to run LSTM /GRU on batch or for loop basis')    

parser.add_argument('-ONLY_GNN', action='store_true',
                    help='to run LSTM /GRU on batch or for loop basis')  

parser.add_argument('-PAGE_RANK', action='store_true',
                    help='to run LSTM /GRU on batch or for loop basis')                    
parser.add_argument('--sample_rate', default= .5,type = float,
                    help='to run LSTM /GRU on batch or for loop basis')   
parser.add_argument('--lr', default = .00001,type = float,
                    help='to run LSTM /GRU on batch or for loop basis')    
parser.add_argument('--lr_decay', default = .9,type = float,
                    help='to run LSTM /GRU on batch or for loop basis')    
parser.add_argument('--boot_sample', default = 10,type = int,
                    help='to run LSTM /GRU on batch or for loop basis') 
parser.add_argument('--early_stop', default = 10,type = int,
                    help='to run LSTM /GRU on batch or for loop basis')                    
parser.add_argument('--repo_path', default = './',type = str,
                    help='to run LSTM /GRU on batch or for loop basis')                    

parser.add_argument('--daset_name', default = 'paper',type = str,
                    help='to run LSTM /GRU on batch or for loop basis')                    


args = parser.parse_args()
print(args)
data_dict,ts_list = get_data_node_classification(args.dataset_name,args.GNN,args.full_batch,page_rank = args.PAGE_RANK,base = args.repo_path)
num_hidden = args.hidden_dim
num_embeddings = args.embedding_dim
n_layers = args.GNN_layers
lr = args.lr
momentum = 0
sample_rate = args.sample_rate
number_feats = data_dict[ts_list[-1]].x.shape[1]
n_nodes =   data_dict[ts_list[0]].x.shape[0]
n_epoch = args.epochs
run = args.run
upsample = True if sample_rate > 0 else False

if args.GNN == 'GAT':
    model_GAT = GAT(number_feats, num_hidden, num_embeddings,edge_dim = data_dict[ts_list[0]].edge_attr.shape[1],heads = 2 ,num_layers=n_layers)
elif args.GNN == 'GCN': 
    model_GAT = GCN(number_feats, num_hidden, num_embeddings,num_layers=n_layers)
if args.RNN == 'GRU':
    model_RNN = GRUClassification(hidden_dim = num_hidden , 
                            input_dim=number_feats if args.ONLY_RNN else num_embeddings, 
                            n_nodes = n_nodes)
elif args.RNN == 'LSTM':
    model_RNN = LSTMClassification(hidden_dim = num_hidden, 
                            input_dim= number_feats if args.ONLY_RNN else num_embeddings, 
                            n_nodes = n_nodes)
model_FFNN = FFNN(hidden_dim=num_hidden, target_size=1)

# model_GAT.to('cuda')
# model_RNN.to('cuda')
# model_FFNN.to('cuda')

print(type(model_GAT))
print(type(model_RNN))

t0 = time.time()

assert not args.ONLY_RNN & args.ONLY_GNN
wind = args.no_windows
if args.ONLY_RNN: 
    
    train_foward_function = train_foward_only_RNN
elif args.ONLY_GNN: 
    assert args.embedding_dim == args.hidden_dim, 'BAD JOB'
    train_foward_function = train_foward_only_GNN
    wind = True

models,metrics = train(data_dict = data_dict,ts_list = ts_list,model_GAT =model_GAT, model_RNN=model_RNN, model_FFNN=model_FFNN,\
n_epoch=n_epoch,lr = lr,momentum = momentum, sample_rate =sample_rate,\
    no_windows=wind,upsample = upsample, train_forward_funcion=train_foward_function, boot_sample = args.boot_sample,early_stop = args.early_stop)


t1 = time.time()
total = t1-t0
print(total)
# Save a trained model to a file.
p = args.repo_path
path = p+ f'/models/{args.run}'

isExist = os.path.exists(path)
log_file = p+'/logs/log_df/logs_hp_new.csv'
rows =[args.run] + [total] + metrics 
df = pd.DataFrame({c:[r] for c,r in zip(range(len(rows)),rows)})
print(df)
with open(log_file, 'a') as f:
    df.to_csv(f, header=False,index = False)

if not isExist:
    os.makedirs(path)
[torch.save(model.state_dict(), path + f'/{args.run}_{name}.pt') for model,name in zip(models,['model_FFNN','model_GAT','model_RNN'])] 





