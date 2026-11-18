import networkx as nx
from ast import parse
import torch
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import Data
import json
import os 
import numpy as np
import pandas as pd
from pathlib import Path
import argparse


def preprocess(data_name):
  u_list, i_list, ts_list, label_list = [], [], [], []
  feat_l = []
  idx_list = []

  with open(data_name) as f:
    s = next(f)
    for idx, line in enumerate(f):
      e = line.strip().split(',')
      #user id 
      u = int(e[0])
      #item id 
      i = int(e[1])
      #time stamp 
      ts = float(e[2])
      #state label/ label of what it is 
      label = float(e[3])  # int(e[3])
      #csv of features anything after index 4 
      #note features must be numeric so we would 1 hot encode categorical variables
      feat = np.array([float(x) for x in e[4:]])

      u_list.append(u)
      i_list.append(i)
      ts_list.append(ts)
      label_list.append(label)
      idx_list.append(idx)

      feat_l.append(feat)
  #returns pandas dataframe with users, ids, time stamps, numpy array of features
  return pd.DataFrame({'u': u_list,
                       'i': i_list,
                       'ts': ts_list,
                       'label': label_list,
                       'idx': idx_list}), np.array(feat_l)


def reindex(df, bipartite=True):
  new_df = df.copy()
  if bipartite:
    assert (df.u.max() - df.u.min() + 1 == len(df.u.unique()))
    assert (df.i.max() - df.i.min() + 1 == len(df.i.unique()))

    upper_u = df.u.max() + 1
    new_i = df.i + upper_u

    new_df.i = new_i
    new_df.u += 1
    new_df.i += 1
    new_df.idx += 1
  else:
    new_df.u += 1
    new_df.i += 1
    new_df.idx += 1
  return new_df

def make_labels(df,label_arr,ts): 
    mask = df[df.label == 1].u.values      
    label_arr[mask,0,int(ts)] = 1 

    return label_arr

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

def run(data_name, page_rank,GCN,val_start ,test_start, bipartite=False):
    base = '/home/' + os.environ.get('USER') + '/projects/def-cbravo/rappi_data/QTR/'
    Path("data/").mkdir(parents=True, exist_ok=True)
    PATH =  base + '/{}.csv'.format(data_name)


    df, feat = preprocess(PATH)
    new_df = reindex(df, bipartite)

    empty = np.zeros(feat.shape[1])[np.newaxis, :]
    feat = np.vstack([empty, feat])
    sub_set = pd.read_csv(base + 'TGN_fin.csv')
    numerical = sub_set.drop(columns = ['APPLICATION_USER_ID','START_DATE']).columns
    keys = sub_set[['START_DATE','APPLICATION_USER_ID']]

    train = sub_set[(val_start > sub_set.START_DATE) ] 
    sub_train = train.drop(columns=['START_DATE','APPLICATION_USER_ID'])

    min_ = sub_train.min()
    max_ = sub_train.max() 

    train_feats = (sub_train - min_) / (max_ - min_)

    val = sub_set[(val_start <= sub_set.START_DATE) & (test_start > sub_set.START_DATE)]
    sub_val = val.drop(columns=['START_DATE','APPLICATION_USER_ID'])

    val_feats = (sub_val - min_) / (max_ - min_)
    val_feats.iloc[:,:] = np.where(val_feats > 1 , 1 , val_feats)
    val_feats.iloc[:,:] = np.where(val_feats < 0 , 0 , val_feats)

    # val_feats.apply(lambda x : 1 if x > 1 else x)

    test = sub_set[(test_start <= sub_set.START_DATE) ]
    sub_test= test.drop(columns=['START_DATE','APPLICATION_USER_ID'])

    test_feats = (sub_test - min_) / (max_ - min_)

    test_feats.iloc[:,:] = np.where(test_feats > 1 , 1 , test_feats)
    test_feats.iloc[:,:] = np.where(test_feats < 0 , 0 , test_feats)


    sub_set[numerical] = pd.concat([train_feats,val_feats,test_feats])

    # feats = (feats- feats.min()) / (feats.max() - feats.min())
    sub_set = sub_set.drop(columns = ['START_DATE','APPLICATION_USER_ID'])
    sub_set = pd.concat([keys,sub_set],axis = 1)
    s = sub_set[sub_set.START_DATE == max(sub_set.START_DATE)].shape
    out = np.zeros((sub_set.APPLICATION_USER_ID.max() + 2,s[1]-2,max(sub_set.START_DATE)+1))
    for ts in sub_set.START_DATE.unique():
        timed = sub_set[sub_set.START_DATE == ts]
        for user in timed.APPLICATION_USER_ID: 

            out[user+1,:,ts] = timed[user == timed.APPLICATION_USER_ID].drop(columns = ['APPLICATION_USER_ID','START_DATE'])


    graph_df = new_df
    edge_features = feat
    inds = np.argwhere(np.sum(edge_features,axis = 1) ==0 ).flatten()
    #equals 1 due to added edge to match node indexing
    assert len(inds.flatten()) == 1, 'EDGES FROM REFERALS IN DATASET'

    node_features = out


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
        edges = np.array([np.concatenate((sources, destinations), axis = None),np.concatenate((destinations,sources), axis = None)])
        edges_pagerank = np.array([sources,destinations])
        labels = make_labels(sub_graph,label_arr,ts)
        nodes = list(set(edges.flatten()) )
        if page_rank:
            n_feats = make_page_rank(edges_pagerank,node_features[:,:,int(ts)])
        else:
            n_feats = node_features[:,:,int(ts)]
 
        n_feats = node_features[:,:,int(ts)]
        print(len(labels),'labels')
        #assert (edges.shape[1] == len(sub_graph.index)), 'MISMATCH IN EDGE FEATS AND EDGE SIZE '
        if GCN == 'GAT': 
            datas[ts]  = Data(n_feats,edges,y = labels[:,:,int(ts)].flatten(),edge_attr = np.concatenate((edge_features[sub_graph.index,:],edge_features[sub_graph.index,:])),nodes = nodes )
        elif GCN == 'GCN': 
            datas[ts]  = Data(n_feats ,edges,y = labels[:,:,int(ts)].flatten(),edge_attr = np.sum(np.concatenate((edge_features[sub_graph.index,:],edge_features[sub_graph.index,:])),axis = 1),nodes = nodes)
        sum_rows+= edges.shape[1]
        print(f'THE MONTH {ts} has {len(edges.flatten())} edges \n it has {len(set(edges.flatten()))} nodes with {len(np.argwhere(labels[:,:,int(ts)].flatten() == 1).flatten())} postives')
        print(f'NODE FEATRUE MATRIX IS DIMENSIONS {n_feats.shape}')
    #assert sum_rows == len(graph_df[graph_df.ts.isin(ts_iter)]), 'NOT SAME NODES AS IN EDGES'
    if page_rank: GCN = 'PR'
    torch.save( datas,'./data/' + f'{data_name}_{GCN}.pt')
    print('SAVING COMPLETE')
    print('SAVED' ,  './data/' + f'{data_name}_{GCN}.pt')

parser = argparse.ArgumentParser('Interface for TGN data preprocessing')
parser.add_argument('--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                        default='wikipedia')
parser.add_argument('--bipartite', action='store_true', help='Whether the graph is bipartite')
parser.add_argument('--page_rank', action='store_true', help='Whether the graph is bipartite')
parser.add_argument('--GNN', type=str, help='Dataset name (eg. wikipedia or reddit)',default ='GCN')
parser.add_argument('--val_start', type=int, help='Dataset name (eg. wikipedia or reddit)',default =13)
parser.add_argument('--test_start', type=int, help='Dataset name (eg. wikipedia or reddit)',default =15)

args = parser.parse_args()

run(args.data, args.page_rank,args.GNN,args.val_start,args.test_start)