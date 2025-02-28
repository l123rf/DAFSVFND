import torch
import torch.nn as nn

class TransformerEncoders(nn.Module):
    def __init__(self, layer_num, d_model, nhead):
        super(TransformerEncoders, self).__init__()
        
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
            for _ in range(layer_num)
        ])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)  
        return x
