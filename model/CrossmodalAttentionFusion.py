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
		"""
		替换原来的前馈网络，使用 1x1 卷积进行特征变换
		"""
		super(ConvFeedForward, self).__init__()
		self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
		self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
		self.gelu = nn.GELU()
		self.dropout = nn.Dropout(dropout)
		self.batch_norm = nn.BatchNorm1d(d_model)

	def forward(self, x):
		residual = x  # 残差连接
		x = x.transpose(1, 2)  # [batch, seq_len, d_model] → [batch, d_model, seq_len]
		x = self.gelu(self.conv1(x))
		x = self.conv2(x)
		# 再次转置回 [batch, seq_len, d_model]
		x = x.transpose(1, 2)  # [batch, d_model, seq_len] → [batch, seq_len, d_model]

		x = self.dropout(x)  # 添加 Dropout

		# 残差连接：确保 `residual` 和 `x` 的形状一致
		# 将 residual 转置为 [batch, d_model, seq_len]
		residual = residual.transpose(1, 2)

		# 归一化和残差连接
		x = self.batch_norm(residual + x.transpose(1, 2))  # 这里需要将 x 和 residual 调整为 [batch, d_model, seq_len]
		x = x.transpose(1, 2)
		return x


class DualAttentionModule(nn.Module):
	def __init__(self, d_model, n_heads, d_ff, dropout):
		"""
		融合注意力模块，实现多模态特征的交互和融合。
		"""
		super(DualAttentionModule, self).__init__()
		self.text_visual_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
		self.text_audio_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
		self.fc = nn.Sequential(torch.nn.Linear(3 * d_model, d_model), torch.nn.ReLU(),nn.Dropout(p=dropout))

		self.layer_norm = nn.LayerNorm(d_model)
		self.feed_forward = ConvFeedForward(d_model, d_ff, dropout)

	def forward(self, F_T, F_V, F_A):
		#学习文本、视觉、音频之间的相互关系，而不仅仅是 attn_output_V 和 attn_output_A 之间的关系，让网络学习到新模态的信息，同时保留文本的核心语义，如果不拼接 F_T，则 fc 只能学习视觉和音频，而无法学习文本对它们的影响。
		# F_T: 文本特征, F_V: 视觉特征, F_A: 音频特征
		attn_output_V, _ = self.text_visual_attention(F_T, F_V, F_V)
		attn_output_A, _ = self.text_audio_attention(F_T, F_A, F_A)

		# 拼接融合
		fused_features = torch.cat([ attn_output_V, F_T,attn_output_A], dim=-1)
		fused_features = self.fc(fused_features)
        

		# 残差连接 + 前馈网络
		fused_features = self.layer_norm(F_T + fused_features)
		fused_features = self.feed_forward(fused_features)
		return fused_features

"""
堆叠多层融合注意力模块，实现更深层次的特征交互。
"""


class Feature_Fusion(nn.Module):
	def __init__(self, d_model, n_heads, d_ff, dropout, num_layers,out_dim):
		"""
		堆叠多层融合注意力模块，实现更深层次的特征交互。
		"""
		super(Feature_Fusion, self).__init__()
		self.layers = nn.ModuleList([
			DualAttentionModule(d_model,n_heads, d_ff, dropout) for _ in range(num_layers)
		])

	def forward(self, F_T, F_V, F_A):

		for layer in self.layers:
			F_T = layer(F_T, F_V, F_A)

		return F_T

