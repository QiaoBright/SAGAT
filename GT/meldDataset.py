import pickle
import torch
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MELDDataset(torch.utils.data.Dataset):
    def __init__(self, train=True, n_classes=7):
        path = 'dataset/meld_features_emoberta.pkl'

        if n_classes == 7:
            self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
            self.videoAudio, self.videoVisual,self.videoSentence, self.trainVid,\
            self.testVid, _ , self.textFea = pickle.load(open(path, 'rb'))
        '''
        label index mapping = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger':6}
        去看具体的self.*****指的是什么 在2-29的enamine_datasets里
        '''
        self.keys = [x for x in (self.trainVid if train else self.testVid)] #[i for i in range(1432)]
        self.len = len(self.keys)
    def __getitem__(self, index):                  # 这个方法主要用于实现类的对象可以通过索引访问，就像列表或数组那样。
        vid = self.keys[index]
        labels = self.videoLabels[vid]
        labels = torch.tensor(labels).to(device)
        videoAudio = torch.from_numpy(self.videoAudio[vid]).to(device)
        videoVisual = torch.from_numpy(self.videoVisual[vid]).to(device)
        sentence_vectors = self.textFea[vid].to(device) #shape形状[句子个数，768]
        speaker_vectors = self.videoSpeakers[vid]    #[[0，1，0，0，0]*句子个数]
        edge_idx_other = self.local_context(sentence_vectors).to(device)
        edge_idx_speaker = self.Independent_short_term(speaker_vectors).to(device)
        edge_idx_longdis = self.Independent_period(speaker_vectors).to(device)

        inter_modal_edges = self.create_inter_modal_edges(sentence_vectors, videoAudio, videoVisual).to(device)
        # return videoAudio, videoVisual, sentence_vectors, edge_idx_other, labels, edge_idx_speaker, edge_idx_longdis
        return videoAudio,videoVisual,sentence_vectors, edge_idx_other, labels, edge_idx_speaker, edge_idx_longdis,inter_modal_edges

    def local_context(self,sentence_vectors):
        """
        计算句子向量列表的边索引。

        参数：(N,768)
            sentence_vectors (torch.Tensor): 形状为 (N, d) 的张量，其中 N 是句子数量，d 是句子向量的维度。

        返回：
            torch.LongTensor: 句子之间的边索引，形状为 (2, num_edges)，其中 num_edges 是边的数量。
        """
        num_sentences = sentence_vectors.size(0)
        edge_index = [[i,i] for i in range(num_sentences)]


        # # 添加每句与前一句之间的边
        for i in range(1, num_sentences-1):
            edge_index.append([i, i-1])
            edge_index.append([i,i+1])



        return torch.LongTensor(edge_index).t().contiguous()

    def Independent_short_term(self,speakers):
        edge_index = [[i,i] for i in range(len(speakers))]
        for i,speaker in enumerate(speakers):
          for j in range(i-1,-1,-1):
            if speaker == speakers[j]:
              edge_index.append([i,j])
              break
        return torch.LongTensor(edge_index).t().contiguous()

    def Independent_period(self,speakers):
      edge_index = [[i,i] for i in range(len(speakers))]
      for i,speaker in enumerate(speakers):
          for j in range(len(speakers)):
            if speaker == speakers[j]:
              edge_index.append([i,j])
      return torch.LongTensor(edge_index).t().contiguous()

    def create_inter_modal_edges(self,sentence_vectors, videoAudio, videoVisual):
        """
        创建跨模态（文本、音频、视觉）的边。

        参数：
            sentence_vectors (torch.Tensor): 文本特征向量，形状为 (N, d_text)。
            videoAudio (torch.Tensor): 音频特征向量，形状为 (N, d_audio)。
            videoVisual (torch.Tensor): 视觉特征向量，形状为 (N, d_visual)。

        返回：
            torch.LongTensor: 跨模态边的索引，形状为 (2, num_edges)。
        """
        num_sentences = sentence_vectors.size(0)
        num_video_features = videoAudio.size(0)

        # 假设每个模态有相同数量的特征点
        assert num_sentences == num_video_features

        # 创建一个空的边索引列表
        inter_modal_edge_index = []

        # 对于每一个句子特征，都尝试连接对应的音频和视觉特征
        for i in range(num_sentences):
            inter_modal_edge_index.extend([
                [i, num_sentences + i],  # 文本 -> 音频
                [num_sentences + i, i],  # 音频 -> 文本
                [i, 2 * num_sentences + i],  # 文本 -> 视觉
                [2 * num_sentences + i, i],  # 视觉 -> 文本
                [num_sentences + i, 2 * num_sentences + i],  # 音频 -> 视觉
                [2 * num_sentences + i, num_sentences + i]  # 视觉 -> 音频
            ])
        return torch.LongTensor(inter_modal_edge_index).t().contiguous()

    def __len__(self):
        return self.len