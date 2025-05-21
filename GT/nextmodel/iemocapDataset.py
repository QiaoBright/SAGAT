import pickle
import torch
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class IEMOCAPDataset(torch.utils.data.Dataset):
    def __init__(self, train=True, n_classes=7):
        path = '../dataset/IEMOCAP_feature_emoberta.pkl'
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        self.testVid, self.textFea = pickle.load(open(path, 'rb'), encoding="latin1")
        '''
        label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
        '''

        self.keys = [x for x in (self.trainVid if train else self.testVid)]
        self.len = len(self.keys)
        # self.len = 5
    def __getitem__(self, index):
        vid = self.keys[index]
        labels = self.videoLabels[vid]
        labels = torch.tensor(labels).to(device)
        videoAudio = torch.tensor(self.videoAudio[vid]).to(device)
        videoVisual = torch.tensor(self.videoVisual[vid]).to(device)
        sentence_vectors = self.textFea[vid].to(device) #shape形状[句子个数，768]
        speaker_vectors = self.videoSpeakers[vid]    #[[0，1，0，0，0]*句子个数]

        return videoAudio, videoVisual, sentence_vectors, speaker_vectors,labels


    def __len__(self):
        return self.len