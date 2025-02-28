import torch
from utils.tools import *
from model.CrossmodalAttentionFusion import *
from model.TimingEncoder import *
from model.FeatureEncoder import *


class classifier(nn.Module):
    def __init__(self, fea_dim, dropout_probability):
        super(classifier, self).__init__()
        self.class_net = nn.Sequential(
            nn.Linear(fea_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=dropout_probability),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=dropout_probability),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=dropout_probability),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=dropout_probability),
            nn.Linear(32, 2)
        )

    def forward(self, fea):
        out = self.class_net(fea)
        return out
    

class DAFSVFNDModel(torch.nn.Module):
    def __init__(self, fea_dim, dropout, dataset):
        super(DAFSVFNDModel, self).__init__()
        if dataset == 'fakesv':
            self.bert = pretrain_bert_wwm_model()
            self.text_dim = 1024
        else:
            self.bert = pretrain_bert_uncased_model()
            self.text_dim = 768
            self.text_linear = nn.Sequential(torch.nn.Linear(self.text_dim, 1024),torch.nn.ReLU(),nn.Dropout(p=dropout))

        self.img_dim = 1024
        self.audio_dim = 1024

        self.num_heads = 8
        self.dropout = dropout

        self.dim = 1024
        self.d_ff = 2048
        self.num_layers_fu = 3
        self.num_layers_trm = 3
        
        self.positional_encoding = PositionalEncoding(d_model = self.dim,dropout = self.dropout)
        
        self.trm = TransformerEncoders(layer_num=self.num_layers_trm, d_model=self.dim, nhead=self.num_heads)
        
        self.Feature_Fusion = Feature_Fusion(d_model= self.dim, n_heads = self.num_heads, d_ff = self.d_ff, dropout = self.dropout, num_layers = self.num_layers_fu ,out_dim = self.dim)
        
        
        self.classifier = classifier(fea_dim=self.dim, dropout_probability=self.dropout)


    def forward(self, **kwargs):
        
        
        ####sem###
        ### Title ###
        title_inputid = kwargs['title_inputid']  # (batch,512,D)
        title_mask = kwargs['title_mask']  
        fea_text = self.bert(title_inputid, attention_mask=title_mask)['last_hidden_state']  
        if self.text_dim == 768:
            fea_text = self.text_linear(fea_text)
        
        ### Audio Frames ###
        fea_audio = kwargs['audio_feas']  # (B,L,D)

        
        ### Image Frames ###
        frames = kwargs['frames']  # (B,L,D)
        
        
        fea_text = self.positional_encoding(fea_text)
        fea_audio = self.positional_encoding(fea_audio)
        fea_img = self.positional_encoding(frames)
        
        
        fea_text = self.trm(fea_text)
        fea_audio = self.trm(fea_audio)
        fea_img = self.trm(fea_img)

        fea = self.Feature_Fusion(fea_text,fea_img,fea_audio)
        
        fea = torch.mean(fea, dim=1)
        
        output = self.classifier(fea)
        
        
        return output,fea
