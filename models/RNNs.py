import torch.nn as nn 
import torch 
from torch.nn import BCEWithLogitsLoss, GRUCell ,LSTMCell
class RNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self):
        pass 
    def __init__hidd():
        pass
    
class LSTMs(RNN):
    def __init__(self,input_dim, hidden_dim, n_layers,n_nodes ) -> None:
        super().__init__()
        c_0 = LSTMCell( input_dim,hidden_dim)
        self.cells = nn.ModuleList([LSTMCell( hidden_dim,hidden_dim) for _ in range(n_layers-1)])
        self.cells.insert(0,c_0)
        self.n_layers = n_layers
        self.input_dim = input_dim 
        self.hidden_dim = hidden_dim
        self.n_nodes = n_nodes
    def forward(self,inps, h0_list):
        prev_h, _ = self.cells[0](inps, h0_list[0])
        h_list= []
        h_list.append((prev_h,_))  

        for i,(l,h_c) in enumerate(zip(self.cells,h0_list)):
            if i == 0 : continue
            (prev_h,c) = l(prev_h, h_c)
            h_list.append((prev_h,c))   
    
        return h_list

    def init__hidd(self): 
        
        return [( torch.ones(self.n_nodes ,self.hidden_dim), torch.ones(self.n_nodes ,self.hidden_dim))  for _ in range(self.n_layers)]

class GRUs(RNN):
    def __init__(self,input_dim, hidden_dim, n_layers ,n_nodes) -> None:
        super().__init__()
        c_0 = GRUCell( input_dim,hidden_dim)
        self.cells = nn.ModuleList([GRUCell( hidden_dim,hidden_dim) for _ in range(n_layers-1)])
        self.cells.insert(0,c_0)
        self.n_layers = n_layers
        self.input_dim = input_dim 
        self.hidden_dim = hidden_dim
        self.n_nodes= n_nodes

    def forward(self,inps, h0_list):

        prev_h = self.cells[0](inps, h0_list[0])

        h_list= []
        h_list.append(prev_h)  
        for i,(l,h_c) in enumerate(zip(self.cells,h0_list)):
            if i == 0 : continue
            prev_h = l(prev_h, h_c)
            h_list.append((prev_h))   
    
        return h_list

    def init__hidd(self): 
        h0s = [ torch.ones(self.n_nodes ,self.hidden_dim)  for _ in range(self.n_layers)]

        return h0s


class LSTMClassification(torch.nn.Module):

    def __init__(self,  input_dim,hidden_dim,n_nodes, **kwargs):
        super(LSTMClassification, self).__init__()
        print( input_dim,hidden_dim,n_nodes)
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.n_nodes = n_nodes
        self.lstm = LSTMCell( input_dim,hidden_dim)


    def forward(self, input_, h0):
        (h, c) = self.lstm(input_, h0[0])

        return [(h,c)]
    def init__hidd( self):
        h0 = torch.randn(self.n_nodes ,self.hidden_dim)
        c0 = torch.randn(self.n_nodes, self.hidden_dim)

        return [(h0,c0)]

class GRUClassification(torch.nn.Module):

    def __init__(self, input_dim,hidden_dim,n_nodes, **kwargs):
        super(GRUClassification, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_nodes = n_nodes
        self.input_dim = input_dim
        self.gru = GRUCell(input_size =  input_dim,hidden_size = hidden_dim)

    def forward(self, input_, h0):
        # print(input_,h0[0])
        h = self.gru(input_, h0[0])
     

        return [h]
    def init__hidd( self):
        h0 = torch.randn(self.n_nodes,self.hidden_dim)
   
        return [h0]

def get_RNN(rnn ):
    if rnn == 'LSTM':
        return LSTMs
    if rnn == 'single_gru':
        return GRUClassification
    if rnn == 'single_lstm':
        return LSTMClassification
    else:
        return GRUs
    
