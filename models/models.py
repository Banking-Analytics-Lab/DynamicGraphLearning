from pprint import pprint
from re import search
from turtle import forward, hideturtle
from typing import Tuple
from utils.utils import upsample_embeddings
import torch 
import torch.nn as nn
from models.GNNs import get_GNN
from models.RNNs import get_RNN
from models.decoder import get_decoder

def get_hidden(h):
    if isinstance(h,Tuple):
        return (h[0].clone().detach(),h[1].clone().detach())
    else: 
        return h.clone().detach()

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward():
        pass 

class RNN_GNN(Model): 
    def __init__(self,upsample,n_nodes, RNN, GNN,DECODER , gnn_input_dim,rnn_input_dim, gnn_embedding_dim,rnn_hidden_dim , gnn_output_dim,rnn_layers,gnn_layers,heads, dropout_rate, edge_dim,train_eps,eps,search_depth,**kwargs  ) -> None:
        super().__init__()
        rnn_kw = { 
            'hidden_dim' : rnn_hidden_dim,
            'input_dim' : rnn_input_dim,
            'n_layers' : rnn_layers,
            'n_nodes' : n_nodes
        }
        gnn_kw = {
            'embedding_dim': gnn_embedding_dim, 
            'input_dim' : gnn_input_dim,
            'n_layers' : gnn_layers,
            'heads' : heads,
            'dropout_rate': dropout_rate,
            'edge_dim' : edge_dim,
            'output_dim' : gnn_output_dim,
            'train_eps' : train_eps,
            'eps':  eps ,
            'search_depth':search_depth
        }
        self.upsample = upsample
        self.RNN = get_RNN(RNN)(**rnn_kw)
        self.GNN = get_GNN(GNN)(**gnn_kw)
        self.decoder = get_decoder(DECODER)(rnn_kw['hidden_dim'])

    def forward(self, month_list,data_dict,h0 = None,train = True): 
        h0s =   self.RNN.init__hidd() if h0 == None else h0 
        hidden_states = h0s
        for i,m in enumerate(month_list): 
            data = data_dict[m]
            labs = data.y
            emb = self.GNN(torch.Tensor(data.x).float(), torch.Tensor(data.edge_index).type(torch.int64),torch.tensor(data.edge_attr).float())
            hidden_states = self.RNN(torch.Tensor(emb), hidden_states) 
            if i == 0:
                h0 = [get_hidden(hidden_states[0])]
        last_h = hidden_states[-1]
        last_h = last_h[0] if type(last_h) == tuple else last_h
        if self.upsample > 0 and train: 

            last_h,labs,synth_index  = upsample_embeddings(last_h,data.y,data.edge_index,self.upsample)
        else: 
            synth_index = []
        scores = self.decoder(last_h,torch.tensor(data.edge_index))


       

        return scores,torch.Tensor(labs),h0,synth_index

class RNN_only(Model):
    def __init__(self,upsample, n_nodes,RNN,DECODER ,rnn_input_dim,rnn_hidden_dim ,rnn_layers ,**kwargs  ) -> None:
        super().__init__()
        rnn_kw = { 
            'hidden_dim' : rnn_hidden_dim,
            'input_dim' : rnn_input_dim,
            'n_layers' : rnn_layers,
            'n_nodes' : n_nodes
        }

        self.upsample = upsample
        self.RNN = get_RNN(RNN)(**rnn_kw)
        self.decoder = get_decoder(DECODER)(rnn_kw['hidden_dim'])

    def forward(self, month_list,data_dict,h0,train  = False): 
        h0s =   self.RNN.init__hidd() if h0 == None else h0 
        hidden_states = h0s
        for i,m in enumerate(month_list): 
            data = data_dict[m]
            labs = data.y
            hidden_states = self.RNN(torch.Tensor(data.x), (hidden_states
            )) 
            
            if i == 0:
                h0 = [get_hidden(hidden_states[0])]
        last_h = hidden_states[-1]
        last_h = last_h[0] if type(last_h) == tuple else last_h
        if self.upsample > 0 : 
            last_h,labs,synth_index  = upsample_embeddings(last_h,data.y,data.edge_index,self.upsample)
        else: 
            synth_index = []
        scores = self.decoder(last_h,data.edge_index)

        return scores,torch.Tensor(labs),h0,synth_index



class GNN_only(Model): 
    def __init__(self, GNN,DECODER, gnn_input_dim, gnn_embedding_dim , gnn_output_dim,gnn_layers,heads, dropout_rate, edge_dim ,eps ,train_eps,search_depth,**kwargs  ) -> None:
        super().__init__()
        gnn_kw = {
            'embedding_dim': gnn_embedding_dim, 
            'input_dim' : gnn_input_dim,
            'n_layers' : gnn_layers,
            'heads' : heads,
            'dropout_rate': dropout_rate,
            'edge_dim' : edge_dim,
            'output_dim' : gnn_output_dim,
            'train_eps' : train_eps,
            'eps':  eps ,
            'search_depth':search_depth
        }
        self.GNN = get_GNN(GNN)(**gnn_kw)
        self.decoder = get_decoder(DECODER)(gnn_output_dim)

    def forward_call(self, data): 
        labs = data.y
        emb = self.GNN(torch.Tensor(data.x).float(), torch.Tensor(data.edge_index).type(torch.int64),torch.tensor(data.edge_attr).float())
        scores = self.decoder(emb)
        h0 = None
        synth_index = []
        return scores,torch.Tensor(labs),h0,synth_index

    def forward(self,month, data_dict,h0=None,train = False):
        assert type(month) == int, 'CANNOT USE WINDOWS WITH ONLY GNN'
        return self.forward_call(data_dict[month])







def get_model(gnn_kw, rnn_kw,decoder_kw): 
    if gnn_kw['GNN'] and rnn_kw['RNN']: 
        kw = dict(rnn_kw ,**gnn_kw)
        kw = dict(kw,**decoder_kw)
        pprint(kw)
        return RNN_GNN(**kw)
    elif gnn_kw['GNN'] and  not rnn_kw['RNN']: 
        kw = dict(gnn_kw,**decoder_kw)
        pprint(kw)
        return GNN_only(**kw)
    elif not gnn_kw['GNN'] and rnn_kw['RNN']: 
        kw = dict(rnn_kw,**decoder_kw)
        return RNN_only( **kw)
    else: 
        print('SPECIFY A MODEL, RNN AND GNN ARE EMPTY')