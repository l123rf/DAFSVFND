import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import tqdm
from sklearn.metrics import *
from tqdm import tqdm
from transformers import AutoConfig, BertModel
from transformers.models.bert.modeling_bert import BertLayer
from zmq import device


class ConvFeedForward(nn.Module):
	def __init__(self, d_model, d_ff, dropout=0.1):
		super(ConvFeedForward, self).__init__()
		self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
		self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
		self.gelu = nn.GELU()
		self.dropout = nn.Dropout(dropout)
		self.batch_norm = nn.BatchNorm1d(d_model)

	def forward(self, x):
		residual = x
		x = x.transpose(1, 2)
		x = self.gelu(self.conv1(x))
		x = self.conv2(x)

		x = x.transpose(1, 2)

		x = self.dropout(x)

		residual = residual.transpose(1, 2)

		x = self.batch_norm(residual + x.transpose(1, 2))
		x = x.transpose(1, 2)
		return x


class DualAttentionModule(nn.Module):
	def __init__(self, d_model, n_heads, d_ff, dropout):
		super(DualAttentionModule, self).__init__()
		self.text_visual_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
		self.text_audio_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
		self.fc = nn.Sequential(torch.nn.Linear(3 * d_model, d_model), torch.nn.ReLU(),nn.Dropout(p=dropout))

		self.layer_norm = nn.LayerNorm(d_model)
		self.feed_forward = ConvFeedForward(d_model, d_ff, dropout)

	def forward(self, F_T, F_V, F_A):
		# F_T: 文本特征, F_V: 视觉特征, F_A: 音频特征
		attn_output_V, _ = self.text_visual_attention(F_T, F_V, F_V)
		attn_output_A, _ = self.text_audio_attention(F_T, F_A, F_A)

		fused_features = torch.cat([ attn_output_V, F_T,attn_output_A], dim=-1)
		fused_features = self.fc(fused_features)

		fused_features = self.layer_norm(F_T + fused_features)
		fused_features = self.feed_forward(fused_features)
		return fused_features


class Feature_Fusion(nn.Module):
	def __init__(self, d_model, n_heads, d_ff, dropout, num_layers,out_dim):
		super(Feature_Fusion, self).__init__()
		self.layers = nn.ModuleList([
			DualAttentionModule(d_model,n_heads, d_ff, dropout) for _ in range(num_layers)
		])

	def forward(self, F_T, F_V, F_A):

		for layer in self.layers:
			F_T = layer(F_T, F_V, F_A)

		return F_T

