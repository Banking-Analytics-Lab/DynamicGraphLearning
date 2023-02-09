import os 
from copy import deepcopy
from statistics import mean 
import numpy as np
from tqdm import tqdm 
from pprint import pprint
import pandas as pd
import torch 
from torch.optim import Adam
import argparse
from utils.utils import get_loss,get_metrics,get_window,evauluate, print_log, write_log
import wandb
from models.models import get_model 
import numpy as np
import random
torch.manual_seed(1)
np.random.seed(0)


def get_var_name(variable):
    names = []
    for v in variable:
        for name, value in globals().items():
            if value is v:
                names.append(name)
    return names

parser = argparse.ArgumentParser('Training execution')
parser.add_argument('--GNN', type=str, help='GAT, SAGE, GCN GIN',default='')
parser.add_argument('--RNN', type=str, help='RNN, GRU/LSTM',default = '')
parser.add_argument('--DECODER', type=str, help='decoder',default = 'LIN')
parser.add_argument('--gnn_input_dim', type=int, help='input dim for GNN, defaults to number of features',default =29)
parser.add_argument('--gnn_output_dim', type=int, help='output dim of gnn, matches input dim of RNN in combined models',default =200)
parser.add_argument('--embedding_dim', type=int, help='hidden dimension for GNN layers',default =200)
parser.add_argument('--heads', type=int, help='attention heads',default =2)
parser.add_argument('--dropout_rate', default = 0.2, help='Dropout rate',type = float)
parser.add_argument('--RNN_hidden_dim', type=int, help='hidden dim for RNNs',default =200)
parser.add_argument('--RNN_layers', type=int, help='layers for RNN',default =1)
parser.add_argument('--GNN_layers', type=int, help='layers for gnn',default =2)
parser.add_argument('--upsample_rate', type=float, help='upsample rate for embedding upsampleing',default =0)
parser.add_argument('--page_rank', action='store_true', help='Whether the graph is bipartite')
parser.add_argument('--epochs', type=int, help='epochs',default =5)
parser.add_argument('--lr', type=float, help='learning rate',default =.0001)
parser.add_argument('--eps', type=float, help='epsilon for GIN',default =0)
parser.add_argument('--boot_sample', type=int, help='sample size for bootstrap',default =10000)
parser.add_argument('--city', type=str, help='data name',default = 'CDMX')
parser.add_argument('--loss', type=str, help='loss name',default = 'bce')
parser.add_argument('--full_windows', action='store_true', help='Use full windows or not')
parser.add_argument('--train_eps', action='store_true', help='Train eps for GIN')
parser.add_argument('--search_depth_SAGE', type=int, help='Depth for SAGE search',default =2)
parser.add_argument('--run_name', type=str, help='Name with combination of hps being ran',default = '')
parser.add_argument('--log_file', type=str, help='Log file',default = 'logs.csv')
args = parser.parse_args()

RNN = args.RNN
GNN = args.GNN
page_rank = args.page_rank
boots = args.boot_sample
if page_rank:
    GNN_type = 'page_rank'
elif not GNN: 
    GNN_type = 'GCN'
else: 
    if GNN == 'GIN' or GNN == 'GAT' or GNN == 'SAGE': 
        GNN_type = 'GAT'
        GNN = 'GAT'
    else: 
        GNN
        GNN_type = 'GCN'

data_path = f'./data/{args.city}_{GNN_type}.pt'

# wandb.init(project="rappi", entity="emilianouw",settings=wandb.Settings(start_method="fork"))
# wandb.init(project="Influencer_detection_final", entity="elena-tiukhova", name =f'{run}')

data_dict = torch.load(data_path) 
ts_list = list(data_dict.keys())
ts_list = [int(x) for x in ts_list]
n_nodes =   data_dict[ts_list[0]].x.shape[0]
n_feats =   data_dict[ts_list[0]].x.shape[1]
rnn_input_dim = args.gnn_output_dim if GNN else n_feats

rnn_kw = { 
    'RNN'  : args.RNN,
    'rnn_input_dim' : rnn_input_dim,
    'rnn_hidden_dim' : args.RNN_hidden_dim,
    'rnn_layers' : args.RNN_layers,
    'upsample' : args.upsample_rate,
    'n_nodes' : n_nodes 
}
gnn_kw ={ 
    'GNN' : args.GNN,
    'gnn_input_dim': args.gnn_input_dim,
    'gnn_embedding_dim' : args.embedding_dim,
    'heads' : args.heads,
    'dropout_rate': args.dropout_rate,
    'edge_dim' :data_dict[ts_list[0]].edge_attr.shape[1] if GNN == 'GAT' else None,
    'gnn_output_dim' : args.gnn_output_dim,
    'gnn_layers': args.GNN_layers,
    'eps': args.eps, 
    'train_eps' : args.train_eps,
    'search_depth': args.search_depth_SAGE
}
decoder_kw = {
    "DECODER": args.DECODER
}
epochs = args.epochs
# other_kw = {'lr': args.lr , 'page_rank':args.page_rank, 'upsample_rate': args.upsample_rate,'epochs':args.epochs }
# kw = {**decoder_kw,**rnn_kw,**gnn_kw,**other_kw}
# wandb.config=kw





windows = get_window(ts_list)


model = get_model(gnn_kw=gnn_kw,rnn_kw=rnn_kw,decoder_kw=decoder_kw)
optimizer  = Adam(model.parameters() , lr = args.lr)
loss_function = get_loss(args.loss)()

# torch.save(model.GNN.state_dict(), './model.pt')

# torch.save(model.RNN.state_dict(), './model_rnn.pt')
# exit(0)

train_losses = []
auc_list = []
auprc_list = []
val_auc_seen = []
val_auc_unseen = []
val_auprc_seen = []
val_auprc_unseen = []
min_val_loss = np.inf
best_epoch = 0 
since_changed = 0
for e in tqdm(range(epochs)): 
    model.train()
    loss = 0 
    running_auprc = []
    running_auc = []
    running_loss = []
    best_e = 0
    h0 = None
    for  month_window in windows:

        optimizer.zero_grad()
        scores_for_loss,labels , h0,synth_index = model(month_window, data_dict,h0) 
        m = month_window if type(month_window) == int else month_window[-1] 
        loss = loss_function(scores_for_loss)
        
        loss.backward()
        optimizer.step()

        running_loss.append(loss.item())

    print(loss.item() , 'epoch: ' ,e )
    model.eval()

    
    train_losses.append(mean(running_loss))

 





write_log(metrics, args.log_file, args.run_name, model)












    



