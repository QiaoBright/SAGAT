import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch.nn import TransformerEncoder, TransformerEncoderLayer


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class GATmodule(nn.Module):
    def __init__(self, in_features, hidden_features1, hidden_features2, out_features, num_heads, dataset, dropout=0.2):
        super(GATmodule, self).__init__()
        self.dataset = dataset
        self.gat1 = GATConv(in_features, hidden_features1, num_heads)
        self.gat2 = GATConv(hidden_features1 * num_heads, hidden_features2, num_heads)
        self.fc_out = {
            'MELD': nn.Linear(hidden_features1 * num_heads, out_features).to(device),
            'IEMOCAP': nn.Linear(hidden_features2 * num_heads, out_features).to(device)
        }
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, edge_index):
        x = F.relu(self.gat1(x, edge_index))

        if self.dataset == 'IEMOCAP':
            x = F.relu(self.gat2(x, edge_index))
        x = self.dropout(x)
        return self.fc_out[self.dataset](x)

class Attention(nn.Module):
    def __init__(self, feature_dim,device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(Attention, self).__init__()
        self.feature_dim = feature_dim
        self.device = device
        self.query = nn.Linear(feature_dim, feature_dim).to(device)
        self.key = nn.Linear(feature_dim, feature_dim).to(device)
        self.value = nn.Linear(feature_dim, feature_dim).to(device)
        self.scale = torch.sqrt(torch.FloatTensor([feature_dim])).to(device)

    def forward(self, query, key, value):
        # Move input tensors to the correct device
        query = query.to(self.device)
        key = key.to(self.device)
        value = value.to(self.device)

        if query.dim() == 2:
            query = query.unsqueeze(1)  # (batch_size, 1, feature_dim)
            key = key.unsqueeze(1)  # (batch_size, 1, feature_dim)
            value = value.unsqueeze(1)  # (batch_size, 1, feature_dim)

        # Apply linear transformations to query, key, and value
        query = self.query(query)  # (batch_size, seq_len, feature_dim)
        key = self.key(key)  # (batch_size, seq_len, feature_dim)
        value = self.value(value)  # (batch_size, seq_len, feature_dim)

        # Compute attention scores
        batch_size, seq_len, _ = query.size()
        scores = torch.bmm(query, key.transpose(1, 2)) / self.scale  # (batch_size, seq_len, seq_len)
        attention_weights = F.softmax(scores, dim=-1)  # (batch_size, seq_len, seq_len)

        # Apply attention weights to values
        output = torch.bmm(attention_weights, value)  # (batch_size, seq_len, feature_dim)
        return output


class TransformerModule(nn.Module):
    def __init__(self, feature_dim, num_heads, num_layers, dropout=0.1):
        super(TransformerModule, self).__init__()
        self.embedding = nn.Linear(feature_dim, 21)
        encoder_layers = TransformerEncoderLayer(d_model=21, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc_out = nn.Linear(21, feature_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = self.fc_out(x)
        return x

class lineConGraph(nn.Module):
    def __init__(self, args):
        super(lineConGraph, self).__init__()
        self.text_lconv = GATmodule(args.in_features, args.hidden_features1,args.hidden_features2, args.out_features, args.num_heads,args.dataset)
        self.text_sconv = GATmodule(args.in_features, args.hidden_features1,args.hidden_features2, args.out_features, args.num_heads,args.dataset)
        self.text_cconv = GATmodule(args.in_features, args.hidden_features1, args.hidden_features2, args.out_features, args.num_heads,args.dataset)

        self.audio_lconv = GATmodule(args.audio_features, args.hidden_features1,args.hidden_features2, args.out_features, args.num_heads,args.dataset)
        self.audio_sconv = GATmodule(args.audio_features, args.hidden_features1,args.hidden_features2, args.out_features, args.num_heads,args.dataset)
        self.audio_cconv = GATmodule(args.audio_features, args.hidden_features1, args.hidden_features2, args.out_features, args.num_heads,args.dataset)

        self.visual_lonv = GATmodule(args.visual_features, args.hidden_features1,args.hidden_features2, args.out_features, args.num_heads,args.dataset)
        self.visual_sonv = GATmodule(args.visual_features, args.hidden_features1,args.hidden_features2, args.out_features, args.num_heads,args.dataset)
        self.visual_conv = GATmodule(args.visual_features, args.hidden_features1, args.hidden_features2, args.out_features, args.num_heads,args.dataset)

        self.audio_feature_transform = nn.Linear(args.audio_features, args.in_features)
        self.visual_feature_transform = nn.Linear(args.visual_features, args.in_features)
        self.decrease=nn.Linear(args.class_num*3,args.class_num)
        self.Modal_interaction = GATmodule(args.in_features, args.hidden_features1,args.hidden_features2, args.out_features, args.num_heads,args.dataset)


        self.attention = Attention(args.out_features*4)

        self.transformer = TransformerModule(args.out_features*1, num_heads=1, num_layers=1)

        self.MLP = nn.Sequential(
            # nn.Linear(args.out_features*3, args.class_num),
            nn.Linear(args.out_features*4, args.class_num),
            nn.Dropout(0.2),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x, x1,x2,edge_index_other, edge_idx_speaker,edge_idx_longdis,modal_interaction):
    # def forward(self, x, x1, x2, edge_index_other, edge_idx_speaker, edge_idx_longdis):
        text_lconv = self.text_lconv(x, edge_idx_speaker)
        text_cconv = self.text_cconv(x, edge_index_other) + self.text_cconv(x, edge_idx_speaker)
        text_sconv = self.text_sconv(x, edge_idx_longdis)

        audio_lconv = self.audio_lconv(x1, edge_idx_speaker)
        audio_cconv = self.audio_cconv(x1, edge_index_other) + self.audio_cconv(x1, edge_idx_speaker)
        audio_sconv = self.audio_sconv(x1, edge_idx_longdis)

        visual_lconv = self.visual_lonv(x2, edge_idx_speaker)
        visual_cconv = self.visual_conv(x2, edge_index_other) + self.visual_conv(x2, edge_idx_speaker)
        visual_sconv = self.visual_sonv(x2, edge_idx_longdis)

        x1 = self.audio_feature_transform(x1)
        x2 = self.visual_feature_transform(x2)
        all_features = torch.cat([x, x1, x2], dim=0)
        Modal_interaction = self.Modal_interaction(all_features,modal_interaction)
        if Modal_interaction.shape[1]==6:
            Modal_interaction = Modal_interaction.reshape(-1,18)
        else:
            Modal_interaction = Modal_interaction.reshape(-1, 21)
        Modal_interaction = self.decrease(Modal_interaction)


        lconv = torch.cat([audio_lconv,visual_lconv,text_lconv,Modal_interaction], dim=1)
        lconv_features = self.attention(lconv,lconv,lconv)


        cconv = torch.cat([audio_cconv,visual_cconv,text_cconv,Modal_interaction], dim=1)
        cconv_features = self.attention(cconv,cconv,cconv)

        sconv = torch.cat([audio_sconv,visual_sconv,text_sconv,Modal_interaction],dim=1)
        sconv_features = self.attention(sconv,sconv,sconv)


        features = (lconv_features+cconv_features+sconv_features).squeeze(1)



        output = self.MLP(features)
        return output