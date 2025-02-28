import torch
import torch.nn as nn

class TransformerEncoders(nn.Module):
    def __init__(self, layer_num, d_model, nhead):
        super(TransformerEncoders, self).__init__()
        
        # 创建n个TransformerEncoderLayer
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
            for _ in range(layer_num)
        ])
        
        # LayerNorm 防止梯度消失或爆炸
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # x的形状应该是 (batch_size, seq_len, d_model)
        for layer in self.layers:
            x = layer(x)  
        return x