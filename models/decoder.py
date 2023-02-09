import torch
from torch_geometric.utils import negative_sampling
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn.models.autoencoder import InnerProductDecoder
class Decoder(nn.Module):
    def __init__(self,hidden_dim=None, target_size=1) -> None:
        super().__init__()

    def forward():
        pass 
    
class LinFFN(Decoder):
    def __init__(self,hidden_dim, target_size=1):
        super().__init__()

        self.fc = Linear(hidden_dim, int(hidden_dim/2))
        self.fc2 = Linear(int(hidden_dim/2), target_size)


    def forward(self, input_,edge_index = None ):
        h = self.fc(input_)
        h = F.relu(h)
        h = self.fc2(h)
        return h

class InnerProduct(Decoder): 
    def __init__(self,hidden_dim) -> None:
        super().__init__()
        self.dec = InnerProductDecoder()
    def forward(self,input_,edge_index):
        # torch geometric GAE recon loss implementation 
        EPS = 1e-15 # adding some error to reduce zero division I think
        neg_sample = negative_sampling(edge_index,input_.size(0))
        log_scores = self.dec(input_, edge_index, sigmoid=True) 
        neg_scores = self.dec(input_, neg_sample, sigmoid=True) 

        return neg_scores + log_scores

class BottleNeck(Decoder): 
    def __init__(self,hidden_dim,steps = 4) -> None:
        super().__init__()
       
        self.dec = InnerProductDecoder()
        self.lins = nn.ModuleList()
        prev_dim = int(hidden_dim)
        delta = int(hidden_dim/steps)
        # 300, 300 -> 200, 300 -100, 200 , 100 
        cur_dim = prev_dim
        for i in range(steps): 
            cur_dim -= delta
            self.lins.append( nn.Linear(prev_dim,cur_dim) )
            prev_dim = cur_dim
        for i in range(steps): 
                cur_dim += delta
                self.lins.append( nn.Linear(prev_dim,cur_dim) )
                prev_dim = cur_dim 

        assert cur_dim == hidden_dim, f"Pick a something that is divisible by {hidden_dim} or recon dimension won't match"


     
    def forward(self,input_,edge_index):
        h = input_
        for l in self.lins: 
            h = l(h)
        neg_sample = negative_sampling(edge_index,input_.size(0))

        log_scores = self.dec(h, edge_index, sigmoid=True) 
        neg_scores = self.dec(h, neg_sample, sigmoid=True) 

        return torch.cat(( log_scores,neg_scores))

def get_decoder(decoder): 
    if decoder == 'LIN':
        return LinFFN
    elif decoder == 'InnerProduct': 
        return InnerProduct
    elif decoder == 'BottleNeck': 
        return BottleNeck
    assert True , 'INCORRECT DECODER NAME'