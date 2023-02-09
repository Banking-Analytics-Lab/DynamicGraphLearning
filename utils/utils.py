from xml.sax.handler import property_interning_dict
import pandas as pd 
import numpy as np 
from scipy.spatial.distance import pdist,squareform
import random 
import torch
from sklearn.metrics import auc, roc_curve, classification_report, precision_recall_curve, accuracy_score
from torch.nn import BCEWithLogitsLoss
from statistics import mean
import os

def upsample_embeddings(embed, labels,edges,sample_rate,end_index = None):

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

def recon_loss(scores):
    EPS = 1e-15
    return -torch.log(scores + EPS ).mean()



def  auprc_wrap(labels,scores):
    pres, recall, _ = precision_recall_curve(labels, scores)
    return recall,pres,_


def get_metrics(scores,labels, ): 

    fpr, tpr, thresholds = roc_curve( labels,scores)
    precision, recall, thresholds = precision_recall_curve(labels,scores)

    return auc(fpr, tpr),auc(recall,precision)

def get_loss(loss,**kwargs) : 
    if loss =='bce': 
        return BCEWithLogitsLoss 
    if loss =='recon': 
        return lambda : recon_loss # return identity loss  fun(x) = x, used hen loss is computed in forward pass


def get_window(dates):
    return [[int(dates[i-1]),int(dates[i]),int(dates[i+1])] for i in range( 1,len(dates) -1 )]

def evauluate(model,windows,data_dict,loss_function,train_nodes,unseen_nodes_set,h0,boot =False,boot_size =10000):
    
    losses = []
    auprcs = []
    auprcs_seen= []
    auprcs_unseen=[]
    aucs= []
    aucs_seen= []
    aucs_unseen = []

    for month_window in windows: 
        m = month_window if type(month_window) == int else month_window[-1] 
        nodes_to_eval= list(set(data_dict[m].edge_index.flatten()))
        unseen_nodes = list(set(nodes_to_eval ) & unseen_nodes_set)
        seen_nodes = list(set(nodes_to_eval) & train_nodes)
        scores_for_loss,labels , h0,synth_index = model(month_window, data_dict,h0,train = False) 
        scores_for_loss = torch.Tensor(scores_for_loss.detach().flatten()).float()
        scores_seen = scores_for_loss[seen_nodes]
        labels_seen = labels[seen_nodes]
               
        scores_unseen = scores_for_loss[unseen_nodes]
        labels_unseen = labels[unseen_nodes]
        loss = loss_function(scores_for_loss, labels)

        auc, auprc = get_metrics(torch.sigmoid(scores_for_loss),labels)
        seen_auc, seen_auprc = get_metrics(torch.sigmoid(scores_seen),labels_seen)
        unseen_auc, unseen_auprc = get_metrics(torch.sigmoid(scores_unseen),labels_unseen)

        losses.append(loss)
        auprcs.append(auprc)
        auprcs_seen.append(seen_auprc)
        auprcs_unseen.append(unseen_auprc)
        aucs.append(auc)
        aucs_seen.append(seen_auc)
        aucs_unseen.append(unseen_auc)

        if boot: 
            seen_boot = bootstrap_preds( scores_seen, labels_seen,num_boot = boot_size)
            unseen_boot = bootstrap_preds( scores_unseen, labels_unseen,num_boot =  boot_size)
            d_seen_auc = get_stats(seen_boot, metric = 'seen_auc')
            d_unseen_auc = get_stats(unseen_boot, metric = 'unseen_auc')

            auprc_seen_boot = bootstrap_preds( scores_seen, labels_seen,auprc_wrap,boot_size)
            auprc_unseen_boot = bootstrap_preds( scores_unseen, labels_unseen,auprc_wrap,boot_size)
            d_seen_auprc =get_stats(auprc_seen_boot, metric = 'seen_auprc')
            d_unseen_auprc = get_stats(auprc_unseen_boot, metric = 'unseen_auprc')
            boot_dict = dict(d_seen_auc , **d_unseen_auc)
            boot_dict = dict(boot_dict,**d_seen_auprc) 
            boot_dict = dict(boot_dict,**d_unseen_auprc)
        else:
            boot_dict = None
    

    return np.mean(aucs),np.mean(aucs_seen),np.mean(aucs_unseen),np.mean(auprcs),np.mean(auprcs_seen),np.mean(auprcs_unseen),np.mean(losses),boot_dict,h0



def print_log(names,metrics,e):
    for n,m in zip(names,metrics):
        print(f"{n} = {m} at epoch {e}")

def bootstrap_preds(preds, labs,curve_fun = roc_curve,num_boot = 10000):

    n = len(preds)
    boot_means = np.zeros(num_boot)
    data = pd.DataFrame({'preds':preds.flatten(),'labs':labs.flatten()})

    np.random.seed(0)
    for i in range(num_boot):
        d = data.sample(n, replace=True)

        fpr, tpr, thresholds = curve_fun(d.labs,d.preds)
        
        boot_means[i] = auc(fpr,tpr)

    return boot_means

def get_stats(arr,metric = 'auc'): 
    boot_dict = {
        f'{metric}_boot_mean' : np.mean(arr),
        f'{metric}_boot_std' : np.std(arr),
        f'{metric}_boot_q0' : np.quantile(arr,0),
        f'{metric}_boot_q1' : np.quantile(arr,.25),
        f'{metric}_boot_q2' : np.quantile(arr,.5),
        f'{metric}_boot_q3' : np.quantile(arr,.75),
        f'{metric}_boot_max' : np.quantile(arr,1),
    }
    return boot_dict

def write_log(metrics, log_file, run_name, model):
    p = f'/home/etiukhov/projects/def-cbravo/emiliano/RappiAnomalyDetection/methods/GNN_LSTM'
    log_file = p + '/logs/' + log_file
    metrics.insert(0,run_name) 
    df = pd.DataFrame({c:[r] for c,r in zip(range(len(metrics)),metrics)})
    with open(log_file, 'a') as f:
        df.to_csv(f, header=False,index = False)

    torch.save(model.state_dict(), p + '/model_files/' + run_name)