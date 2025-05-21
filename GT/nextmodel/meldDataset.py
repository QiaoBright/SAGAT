import pickle
import torch
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MELDDataset(torch.utils.data.Dataset):
    def __init__(self, train=True, n_classes=7):
        path = '../dataset/meld_features_emoberta.pkl'

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
        return videoAudio,videoVisual,sentence_vectors, speaker_vectors,labels

    def __len__(self):
        return self.len