import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
	def __init__(self, d_model,dropout, max_len=5000):
		super(PositionalEncoding, self).__init__()
        
		self.dropout = nn.Dropout(p=dropout)
		self.encoding = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
		self.encoding[:, 0::2] = torch.sin(position * div_term)
		self.encoding[:, 1::2] = torch.cos(position * div_term)
		self.encoding = self.encoding.unsqueeze(0)

	def forward(self, x):
		seq_len = x.size(1)
		x = x + self.encoding[:, :seq_len, :].to(x.device)
		return self.dropout(x)