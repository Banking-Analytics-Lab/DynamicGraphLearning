import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
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


    def forward(self, input_):
        h = self.fc(input_)
        h = F.relu(h)
        h = self.fc2(h)
        return h
def get_decoder(decoder): 
    if decoder == 'LIN':
        return LinFFN
    assert True , 'INCORRECT DECODER NAME'