import torch.nn as nn 
import torch 
from torch_geometric.nn import GCNConv, GATv2Conv
from torch.nn import Linear
import torch.nn.functional as F

class GNN(nn.Module):
    def __init__(self,input_dim=None, embedding_dim=None,output_dim=None,edge_dim = None,heads = 8, n_layers = 5) -> None:
        super().__init__()
    def forward(self):
        pass 

class GATs(nn.Module): 
    def __init__(self,input_dim, embedding_dim,output_dim,edge_dim,heads, n_layers, dropout_rate, **kwargs) -> None:
        super().__init__()
        self.dropout_rate = dropout_rate
        self.gat1 = GATv2Conv(input_dim, embedding_dim, heads=heads,edge_dim = edge_dim) # dim_h * num heads
        #dim_h * heads > dim_h
        self.GAT_list = torch.nn.ModuleList([GATv2Conv(embedding_dim*heads, embedding_dim, heads=heads,edge_dim = edge_dim)  for _ in range(n_layers-1)])
        self.gat2 = GATv2Conv(embedding_dim*heads, output_dim, heads=1, edge_dim = edge_dim)
        # for m in self.gat1.parameters():
        #     print(m)
        #     # print(torch.nn.init.xavier_uniform_(m.unsqueeze(0)))


    def forward(self, inp, edge_index,edge_feats):

        h = self.gat1(inp, edge_index,edge_attr = edge_feats)

        # print(self.gat1)
        # print([x for x in self.gat1.parameters()])
        h = F.elu(h)
        h = F.dropout(h, self.dropout_rate)
        for l in self.GAT_list:
            h = l(h, edge_index,edge_attr = edge_feats)
            h = F.elu(h)
            h = F.dropout(h, self.dropout_rate)
        #h = self.lin(h)
        h = self.gat2(h, edge_index, edge_feats)
        h = F.relu(h)
        h = F.dropout(h, self.dropout_rate)
        


        return h
    
class GCNs(GNN):
  """Graph Convolutional Network"""
  def __init__(self, input_dim,embedding_dim, output_dim,n_layers,dropout_rate, **kwargs):
    super().__init__()
    self.gcn1 = GCNConv(input_dim, embedding_dim)
    self.dropout_rate = dropout_rate
    self.GCN_list = torch.nn.ModuleList([GCNConv(embedding_dim, embedding_dim)  for _ in range(n_layers-1)])
    self.lin = Linear(embedding_dim, output_dim)

  def forward(self, x, edge_index,edge_feats):
    #h = F.dropout(x, p=0.5, training=self.training)
    h = self.gcn1(x, edge_index,edge_weight = edge_feats)
    h = F.elu(h)
    h = F.dropout(h, self.dropout_rate)
    for l in self.GCN_list:
        h = l(h, edge_index,edge_weight = edge_feats)
        h = F.elu(h)
        h = F.dropout(h, self.dropout_rate)
    h = self.lin(h)
    h = F.relu(h)
    return h

def get_GNN(gnn ):
    if gnn == 'GAT':
        return GATs
    else: 
        return GCNs