import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch.nn import TransformerEncoder, TransformerEncoderLayer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Unimodal_Encoding_and_Speaker_Embedding(nn.Module):
    def __init__(self, audio_DIMfeatures, visual_DIMfeatures, text_DIMfeatures, num_speakers, dropout=0.2):
        super(Unimodal_Encoding_and_Speaker_Embedding, self).__init__()
        self.fc_audio_features = nn.Linear(audio_DIMfeatures, text_DIMfeatures).to(device)
        self.fc_visual_features = nn.Linear(visual_DIMfeatures, text_DIMfeatures).to(device)
        self.lstm = nn.LSTM(input_size=text_DIMfeatures, hidden_size=128, num_layers=2, batch_first=True)
        self.fc_text_features = nn.Linear(768, text_DIMfeatures).to(device)
        self.speaker_embedding = nn.Embedding(num_speakers, text_DIMfeatures).to(device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, audio_features, visual_features, text_features, speaker_indices):
        speaker_indices = torch.argmax(torch.tensor(speaker_indices), dim=1).to(device)
        speaker_embeddings = self.speaker_embedding(speaker_indices)
        assert audio_features.shape[1] == self.fc_audio_features.in_features, "Audio features dimension mismatch"
        assert visual_features.shape[1] == self.fc_visual_features.in_features, "Visual features dimension mismatch"

        audio_features = self.fc_audio_features(audio_features).unsqueeze(0)
        audio_features, (h_n, c_n) = self.lstm(audio_features)
        visual_features = self.fc_visual_features(visual_features).unsqueeze(0)
        visual_features, (h_n, c_n) = self.lstm(visual_features)
        audio_features, visual_features = audio_features.squeeze(0) + speaker_embeddings, visual_features.squeeze(
            0) + speaker_embeddings

        text_features = self.fc_text_features(text_features)
        A_T = torch.concat((audio_features, text_features), dim=0)
        V_T = torch.concat((visual_features, text_features), dim=0)
        A_V = torch.concat((audio_features, visual_features), dim=0)
        A_A = torch.concat((text_features, text_features), dim=0)

        return A_T, A_V, V_T, A_A


class TransformerModule(nn.Module):
    def __init__(self, feature_dim, num_heads, num_layers, dropout=0.1):
        super(TransformerModule, self).__init__()
        self.embedding = nn.Linear(feature_dim, feature_dim)
        encoder_layers = TransformerEncoderLayer(d_model=feature_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc_out = nn.Linear(feature_dim, feature_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = self.fc_out(x)
        return x


class GATmodule(nn.Module):
    def __init__(self, in_features, hidden_features1, hidden_features2, out_features, num_heads, dataset, dropout=0.2):
        super(GATmodule, self).__init__()
        self.dataset = dataset
        self.gat1 = GATConv(in_features, hidden_features1, num_heads)
        self.gat2 = GATConv(hidden_features1 * num_heads, hidden_features2, num_heads)
        self.fc_out = nn.Linear(hidden_features2 * num_heads, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = F.relu(self.gat1(x, edge_index))
        x = F.relu(self.gat2(x, edge_index))
        x = self.dropout(x)
        return self.fc_out(x)


class Graph_Construction(nn.Module):
    def __init__(self, audio_DIMfeatures, visual_DIMfeatures, text_DIMfeatures, num_speakers, args):
        super(Graph_Construction, self).__init__()
        self.Unimodal_Encoding_and_Speaker_Embedding = Unimodal_Encoding_and_Speaker_Embedding(audio_DIMfeatures,
                                                                                               visual_DIMfeatures,
                                                                                               text_DIMfeatures,
                                                                                               num_speakers).to(device)
        self.TransformerModule = TransformerModule(args.class_num * 3, 1, 2)
        self.A_T_grath = GATmodule(args.in_features, args.hidden_features1, args.hidden_features2, args.out_features,
                                   args.num_heads, args.dataset)
        self.A_V_grath = GATmodule(args.in_features, args.hidden_features1, args.hidden_features2, args.out_features,
                                   args.num_heads, args.dataset)
        self.V_T_grath = GATmodule(args.in_features, args.hidden_features1, args.hidden_features2, args.out_features,
                                   args.num_heads, args.dataset)
        self.MLP = nn.Sequential(
            nn.Linear(args.out_features * 6, args.out_features * 3),
            nn.Linear(args.out_features * 3, args.class_num),
            nn.Dropout(0.2),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, audio_features, visual_features, text_features, speaker_indices):
        A_T, A_V, V_T, A_A = self.Unimodal_Encoding_and_Speaker_Embedding(audio_features, visual_features,
                                                                          text_features,
                                                                          speaker_indices)
        A_T, A_V, V_T, A_A = A_T.to(device), A_V.to(device), V_T.to(device), A_A.to(device)
        Node_Edges1 = self.Node_Edges1(A_T)
        Node_Edges2 = self.Node_Edges2(speaker_indices)
        A_T_grath, A_V_grath, V_T_grath, A_A_grath = self.A_T_grath(A_T, Node_Edges1), self.A_V_grath(A_V,
                                                                                                      Node_Edges1), self.V_T_grath(
            V_T, Node_Edges1), self.A_T_grath(A_A, Node_Edges1)
        A_T_grath1, A_V_grath1, V_T_grath1, A_A_grath1 = self.A_T_grath(A_T, Node_Edges2), self.A_V_grath(A_V,
                                                                                                          Node_Edges2), self.V_T_grath(
            V_T, Node_Edges2), self.A_T_grath(A_A, Node_Edges2)
        doubleA, doubleV, doubleT = (A_T_grath + A_V_grath + A_T_grath1 + A_V_grath1), (A_V_grath + V_T_grath + A_V_grath1 + V_T_grath1), (A_A_grath + A_A_grath1)
        combined_features = torch.cat((doubleA, doubleV, doubleT), dim=1)
        features = self.TransformerModule(combined_features)
        output = self.MLP(features.reshape(-1, features.shape[1] * 2))
        return output

    def Node_Edges1(self, Associative_features):
        num = Associative_features.shape[0]
        edge_index = [[i, i] for i in range(num)]
        if num == 2:
            return torch.LongTensor(edge_index).t().contiguous().to(device)
        if num == 4:
            edge_index += [[0, 1], [0, 2], [1, 0], [1, 3], [2, 3], [2, 0], [3, 1], [3, 2]]
            return torch.LongTensor(edge_index).t().contiguous().to(device)
        for i in range(num):
            if i == 0:
                edge_index += [[0, 1], [0, 2], [0, num // 2]]
            elif i == num - 1:
                edge_index += [[i, i - 1], [i, i - 2], [i, num // 2 - 1]]
            elif i == num // 2 - 1:
                edge_index += [[i, i - 1], [i, i - 2], [i, num - 1]]
            elif i == num // 2:
                edge_index += [[i, i + 1], [i, i + 2], [i, 0]]
            elif i < num // 2 - 1:
                edge_index += [[i, i - 1], [i, i + 1], [i, num // 2 + i]]
            elif i > num // 2:
                edge_index += [[i, i - 1], [i, i + 1], [i, i - num // 2]]
        return torch.LongTensor(edge_index).t().contiguous().to(device)

    def Node_Edges2(self, speaker_indices):
        speakers = [[item.item() for item in sublist] for sublist in speaker_indices]
        num = len(speakers)
        edge_index = [[i, i] for i in range(num)]
        for i, speaker in enumerate(speakers):
            for j in range(len(speakers)):
                if speaker == speakers[j]:
                    edge_index += [[i, j], [j, i]]
                    edge_index += [[i + num // 2, i + num // 2 + j if i + num // 2 + j < num else num - 1],
                                   [i + num // 2 + j if i + num // 2 + j < num else num - 1, i + num // 2]]
                    edge_index += [[i, i + num // 2], [i + num // 2, i]]
        return torch.LongTensor(edge_index).t().contiguous().to(device)
