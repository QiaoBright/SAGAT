import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import argparse
import time,pickle
import torch
import numpy as np
import random
from transformers import AdamW, get_cosine_schedule_with_warmup
from torch.optim import Adam
from tqdm.auto import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch_geometric.nn import GATConv
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#这一行创建了一个ArgumentParser对象，它是argparse模块的主要类，用于管理所有参数的定义和解析。
parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', default=1e-4, type=float, help='Learning Rate')          #定义学习率
parser.add_argument('--warmup_proportion', default=0.1, type=float, help='Set Warmup Proportion')    #定义学习率预热比例 如果设置warmup_proportion为0.1，意味着在训练的前10%的步骤中，学习率会从一个非常小的值（可能是接近于0）逐渐线性增加到其设定的初始值。在预热阶段结束后，学习率保持在初始值，或者按照其他的调度策略（如线性衰减、余弦退火等）进行调整。
parser.add_argument('--num_epochs', default=12, type=int, help='Set Number of Epochs')
parser.add_argument('--batch_size', default=1, type=int, help='Set Batch Size')
parser.add_argument('--mode', default='train', type=str, choices=['train', 'test'], help='Train Or Test Mode')
parser.add_argument('--gradient_accumulation_step', default=8, type=int) # 8               #定义了一个整型参数--gradient_accumulation_step，默认值是8，用于控制梯度累积的步数，这对于在有限的GPU内存下使用更大的有效批量是有用的。
parser.add_argument('--model_output_dir',default='./drive/MyDrive/part2/TestByQiaoFINANCE/',choices=['./drive/MyDrive/part2/TestByQiaoMELD/', './drive/MyDrive/part2/TestByQiaoIEMOCAP/','./drive/MyDrive/part2/TestByQiaoFINANCE/'], type=str)
parser.add_argument('--in_features', type=int, default=768, help='输入特征的维度')
parser.add_argument('--hidden_features1', type=int, default=256, help='隐藏层特征的维度')
parser.add_argument('--hidden_features2', type=int, default=128, help='隐藏层特征的维度')
parser.add_argument('--out_features', type=int, default=7, help='输出特征的维度') # IEMOCAP 6['hap', 'sad', 'neu', 'ang', 'exc', 'fru'] MELD 7['neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger']
parser.add_argument('--num_heads', type=int, default=16, help='GATConv 中的注意力头数') #IEMOCAP 1头好 MELD 4头好
parser.add_argument('--momentum', type=float, default=0.9, help='SGD优化器(随机梯度下降)的动量参数')
parser.add_argument('--dataset', type=str, default='FINANCE', choices=['IEMOCAP', 'MELD','FINANCE'], help='Dataset to use (IEMOCAP or MELD)')
parser.add_argument('--gamma', type=int, default=8, help='focalloss的gamma')
parser.add_argument('--n', type=int, default=8, help='loss的n')
parser.add_argument('--beta', type=float, default=5e-8, help='beta')
parser.add_argument('--theta', type=float, default=0.001, help='theta')

args = parser.parse_args([])
if args.dataset == 'IEMOCAP':
    args.out_features = 6
    args.class_num = 6
    args.model_output_dir = './drive/MyDrive/part2/TestByQiaoIEMOCAP/'
elif args.dataset == 'MELD':
    args.out_features = 7
    args.class_num = 7
    args.model_output_dir = './drive/MyDrive/part2/TestByQiaoMELD/'
elif args.dataset == 'FINANCE':
    args.out_features = 5
    args.class_num = 5
    args.model_output_dir = 'output/finance'
print(args)
class GATmodule(nn.Module):
    def __init__(self, in_features, hidden_features1, hidden_features2, out_features, num_heads, dropout = 0):
        super(GATmodule, self).__init__()
        self.gat1 = GATConv(in_features, hidden_features1, num_heads)
        self.gat2 = GATConv(hidden_features1 * num_heads, hidden_features2, num_heads)
        self.fc_out = nn.Linear(hidden_features2 * num_heads, out_features) # 对多头的输出降维

    def forward(self, x, edge_index):
        x = F.relu(self.gat1(x, edge_index))
        x = F.relu(self.gat2(x, edge_index))
        x = self.fc_out(x)
        return x
class lineConGraph(nn.Module):
    def __init__(self, args):
        super(lineConGraph, self).__init__()
        self.sconv1 = GATmodule(args.in_features, args.hidden_features1,args.hidden_features2, args.out_features, args.num_heads)
        self.sconv2 = GATmodule(args.in_features, args.hidden_features1,args.hidden_features2, args.out_features, args.num_heads)
        self.cconv = GATmodule(args.in_features, args.hidden_features1, args.hidden_features2, args.out_features, args.num_heads)
        self.MLP = nn.Sequential(
            nn.Linear(args.out_features, args.class_num),
            nn.LogSoftmax(dim=1)
        )
    def forward(self, x, edge_index_other, edge_idx_speaker,edge_idx_longdis):
        emb1 = self.sconv1(x, edge_idx_speaker) # Special_GCN out1 -- sadj structure graph
        com1 = self.cconv(x, edge_index_other)  # Common_GCN out1 -- sadj structure graph
        com2 = self.cconv(x, edge_idx_speaker)  # Common_GCN out2 -- fadj feature graph
        emb2 = self.sconv2(x, edge_idx_longdis) # Special_GCN out2 -- fadj feature graph
        output = self.MLP(emb1+emb2+com1+com2)
        return output

class MELDDataset(torch.utils.data.Dataset):
    def __init__(self, train=True, n_classes=7):
        path = './drive/MyDrive/part2/MELD_feature_emoberta.pkl'

        if n_classes == 7:
            self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
            self.videoAudio, self.videoSentence, self.trainVid,\
            self.testVid, _ , self.textFea = pickle.load(open(path, 'rb'))
        '''
        label index mapping = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger':6}
        去看具体的self.*****指的是什么 在2-29的enamine_datasets里
        '''
        self.keys = [x for x in (self.trainVid if train else self.testVid)] #[i for i in range(1432)]
        # print(len(self.keys))
        self.len = len(self.keys)
        # self.len = 1
    def __getitem__(self, index):                  # 这个方法主要用于实现类的对象可以通过索引访问，就像列表或数组那样。
        vid = self.keys[index]
        labels = self.videoLabels[vid]
        labels = torch.tensor(labels).to(device)
        sentence_vectors = self.textFea[vid].to(device) #shape形状[句子个数，768]
        speaker_vectors = self.videoSpeakers[vid]    #[[0，1，0，0，0]*句子个数]
        edge_idx_other = self.local_context(sentence_vectors).to(device)
        edge_idx_speaker = self.Independent_period(speaker_vectors).to(device)
        edge_idx_longdis = self.Independent_short_term(speaker_vectors).to(device)
        # print(edge_idx_other)
        # print(edge_idx_speaker)
        # print(edge_idx_longdis)
        return sentence_vectors, edge_idx_other, labels, edge_idx_speaker, edge_idx_longdis

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
        for i in range(1, num_sentences):
            edge_index.append([i, i-1])
            edge_index.append([i-1, i])
        # 添加每句与后一句之间的边
        for i in range(num_sentences-1):
            edge_index.append([i, i+1])
            edge_index.append([i+1, i])

        return torch.LongTensor(edge_index).t().contiguous()

    def Independent_short_term(self,speakers):
        edge_index = [[i,i] for i in range(len(speakers))]
        for i,speaker in enumerate(speakers):
          for j in range(i-1,-1,-1):
            if speaker == speakers[j]:
              edge_index.append([i,j])
              break

        for i,speaker in enumerate(speakers):
          for j in range(i+1,len(speakers)):
            if speaker == speakers[j]:
              edge_index.append([i,j])
              break
        return torch.LongTensor(edge_index).t().contiguous()

    def Independent_period(self,speakers):
      edge_index = [[i,i] for i in range(len(speakers))]
      for i,speaker in enumerate(speakers):
        for j in range(i-1,-1,-1):
          if speaker == speakers[j]:
            edge_index.append([i,j])

      for i,speaker in enumerate(speakers):
          for j in range(i+1,len(speakers)):
            if speaker == speakers[j]:
              edge_index.append([i,j])
      return torch.LongTensor(edge_index).t().contiguous()

    def __len__(self):
        return self.len

class FinanceDataset(torch.utils.data.Dataset):
    def __init__(self, train=True, n_classes=5):
        path = 'dataset/dialogue_data_long_fenci_tensor.pkl'

        if n_classes == 5:
            self.videoIDs, self.text, self.videoLabels, self.videoSpeakers, self.trainVid, self.textVid, self.sentence_ventors = pickle.load(open(path, 'rb'))
            #self.videoIDs, self.text, self.videoLabels, self.trainVid,__,self.textFea, self.videoSpeakers,self.sentence_ventors,_= pickle.load(open(path, 'rb'))
        '''
        label index mapping = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger':6}
        去看具体的self.*****指的是什么 在2-29的enamine_datasets里
        '''
        # self.textVid=[i for i in range(4805, 6865) if i != 5566]
        self.keys = [x for x in (self.trainVid if train else self.textVid)] #[i for i in range(1432)]
        # print(len(self.keys))
        self.len = len(self.keys)
        # self.len = 1
    def __getitem__(self, index):
        vid = self.keys[index]
        labels = self.videoLabels[vid]
        labels = torch.tensor(labels).to(device)
        sentence_vectors = self.sentence_ventors[vid].to(device) #shape形状[句子个数，768]
        speaker_vectors = self.videoSpeakers[vid]    #[[0，1，0，0，0]*句子个数]
        edge_idx_other = self.local_context(sentence_vectors).to(device)
        edge_idx_speaker = self.Independent_period(speaker_vectors).to(device)
        edge_idx_longdis = self.Independent_short_term(speaker_vectors).to(device)
        return sentence_vectors, edge_idx_other, labels, edge_idx_speaker, edge_idx_longdis

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
        for i in range(1, num_sentences):
            edge_index.append([i, i-1])
            edge_index.append([i-1, i])
        # 添加每句与后一句之间的边
        for i in range(num_sentences-1):
            edge_index.append([i, i+1])
            edge_index.append([i+1, i])

        return torch.LongTensor(edge_index).t().contiguous()

    def Independent_short_term(self,speakers):
        edge_index = [[i,i] for i in range(len(speakers))]
        for i,speaker in enumerate(speakers):
          for j in range(i-1,-1,-1):
            if speaker == speakers[j]:
              edge_index.append([i,j])
              break

        for i,speaker in enumerate(speakers):
          for j in range(i+1,len(speakers)):
            if speaker == speakers[j]:
              edge_index.append([i,j])
              break
        return torch.LongTensor(edge_index).t().contiguous()

    def Independent_period(self,speakers):
      edge_index = [[i,i] for i in range(len(speakers))]
      for i,speaker in enumerate(speakers):
        for j in range(i-1,-1,-1):
          if speaker == speakers[j]:
            edge_index.append([i,j])

      for i,speaker in enumerate(speakers):
          for j in range(i+1,len(speakers)):
            if speaker == speakers[j]:
              edge_index.append([i,j])
      return torch.LongTensor(edge_index).t().contiguous()

    def __len__(self):
        return self.len

class IEMOCAPDataset(torch.utils.data.Dataset):
    def __init__(self, train=True, n_classes=7):
        path = './drive/MyDrive/part2/IEMOCAP_feature_emoberta.pkl'
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        self.testVid, self.textFea = pickle.load(open(path, 'rb'), encoding="latin1")
        '''
        label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
        '''
        # del self.videoVisual['Ses05F_script02_2']
        # del self.videoAudio['Ses05F_script02_2']
        # del self.videoSpeakers['Ses05F_script02_2']
        # del self.videoLabels['Ses05F_script02_2']
        # self.testVid.remove('Ses05F_script02_2')
        self.keys = [x for x in (self.trainVid if train else self.testVid)]
        self.len = len(self.keys)
        # self.len = 5

    def __getitem__(self, index):
        vid = self.keys[index]
        labels = self.videoLabels[vid]
        labels = torch.tensor(labels).to(device)
        sentence_vectors = self.textFea[vid].to(device)
        speaker_vectors = self.videoSpeakers[vid]
        edge_idx_other = self.compute_edge_index_with_past_future(sentence_vectors).to(device)
        edge_idx_speaker = self.build_edge_index_with_same_speaker(speaker_vectors).to(device)
        edge_idx_longdis = self.build_edge_index_with_long_dis(speaker_vectors).to(device)

        return sentence_vectors, edge_idx_other, labels, edge_idx_speaker,edge_idx_longdis

    def compute_edge_index_with_past_future(self,sentence_vectors):
        """
        计算句子向量列表的边索引。

        参数：
            sentence_vectors (torch.Tensor): 形状为 (N, d) 的张量，其中 N 是句子数量，d 是句子向量的维度。

        返回：
            torch.LongTensor: 句子之间的边索引，形状为 (2, num_edges)，其中 num_edges 是边的数量。
        """
        num_sentences = sentence_vectors.size(0)
        edge_index = []

        # 添加每句与前一句之间的边
        for i in range(1, num_sentences):
            edge_index.append([i, i-1])
            edge_index.append([i-1, i])
        # 添加自循环
        for i in range(num_sentences):
            edge_index.append([i, i])


        # # 添加每句与后一句之间的边
        # for i in range(num_sentences - 1):
        #     edge_index.append([i, i+1])
        #     edge_index.append([i+1, i])

        return torch.LongTensor(edge_index).t().contiguous()

    def build_edge_index_with_same_speaker(self, speakers):
        edge_index = []
        num_sentences = len(speakers)
        for i, speaker in enumerate(speakers):
            # 添加与自身的边
            edge_index.append([i, i])
            for j in range(i+1, num_sentences):
                if speakers[j] == speaker:  # 与当前说话人相同的句子建立边
                    edge_index.append([i, j])
        return torch.tensor(edge_index).t().contiguous()

    def build_edge_index_with_long_dis(self, speakers):
        edge_index = []
        num_sentences = len(speakers)
        for i, speaker in enumerate(speakers):
            # 添加与自身的边
            # edge_index.append([i, i])
            for j in range(i, -1, -1):
                if speakers[j] == speaker:  # 与当前说话人相同的前面一个句子建立边
                    edge_index.append([i, j])
                    break
            for j in range(i+1, num_sentences):
                if speakers[j] == speaker:  # 与当前说话人相同的后边一个句子建立边
                    edge_index.append([i, j])
                    break
            # edge_index = remove_self_loops(edge_index)[0]
            # # 然后，添加自环边，以确保每个节点至少有一个自环
            # edge_index, _ = add_self_loops(edge_index, num_nodes=None)
            # # 最后，再次去除重复边
            # edge_index = torch.unique(edge_index, dim=1)
        return torch.tensor(edge_index).t().contiguous()


    def __len__(self):
        return self.len

def set_random_seeds(seed):
    """
    设置随机种子以确保结果的一致性。
    禁用cuDNN的某些特性以提高确定性。
    """
    print("Seed: {}".format(seed))

    # 设置PyTorch的cuDNN相关选项
    torch.backends.cudnn.deterministic = True  # 强制cuDNN使用确定性算法
    torch.backends.cudnn.benchmark = False    # 禁用cuDNN的自动调优
    # torch.backends.cudnn.enabled = False      # 禁用cuDNN

    # 设置Python和NumPy的随机种子
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    # 设置PyTorch的随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU，设置所有GPU的随机种子

def fetch_model_optim_sched(device, args):
    model = lineConGraph(args)
    # 显卡大于1块时，device_ids选择模型载入数据对应的显卡, 但是多卡有不平衡的问题
    # if torch.cuda.device_count() > 1:
    #     model= nn.DataParallel(model,device_ids=[0,1])
    model.to(device)

    # optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)

    '''
    get_cosine_schedule_with_warmup: 这是Hugging Face Transformers库中的一个函数，用于生成一个学习率调度器，它结合了warmup阶段和余弦退火策略。
    optimizer: 这是之前定义的优化器对象，例如SGD、Adam等，它包含了模型的所有可训练参数以及优化算法的超参数（如学习率、动量等）。
    num_warmup_steps: warmup阶段的步数。在训练开始时，学习率会从0线性增长到初始设定的学习率。这有助于避免训练初期梯度爆炸的问题。在这里，warmup步骤的数量被设置为总优化步骤的args.warmup_proportion比例。
    num_training_steps: 总的训练优化步骤数，通常等于训练集的大小除以批量大小，再乘以总的训练轮数（epoch）。这代表了整个训练过程中优化器将会执行梯度下降的次数。
    args.warmup_proportion: 是一个参数，定义了warmup阶段占总训练步骤的比例。例如，如果设置为0.1，则warmup阶段会持续总训练步骤的10%。
    args.num_train_optimization_steps: 这是整个训练过程中计划的优化步骤总数，由用户在训练脚本中预先定义
    '''


    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_proportion * args.num_train_optimization_steps,
        num_training_steps=args.num_train_optimization_steps,
    )

    return model, optimizer, scheduler

from torch.autograd import Variable
class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
                                (默认情况下，损失对每个小批的观测值取平均值。
                                但是，如果字段大小平均值设置为False，则会对每个小批的损失求和)


    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        '''
        这段代码确保self.alpha总是作为一个Variable类型的对象存在，它可以是一个预定义的Variable，
        一个由torch.ones创建并转换成Variable的张量，或者是由其他类型转换而来的Variable。
        '''
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)         #创建和inputs.data类型，形状为(N,C)全0的tensor
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)             #在PyTorch中对张量class_mask进行操作，用于创建one-hot编码向量或矩阵
        #print(class_mask)


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()                      #probs.对数值
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p
        #print('-----bacth_loss------')
        #print(batch_loss)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

def loop(dataloader, model, optimizer, scheduler, device, args, mode='train'):
    loss_cross_entropy_func = F.cross_entropy

    if args.dataset == 'IEMOCAP':
        class_nums = [504, 839, 1324, 933, 742, 1468]
        # class_nums = [135, 229, 379, 168, 295, 358]
    elif args.dataset == 'MELD':
        class_nums = [5180, 1355, 308, 794, 1906, 293, 1262]
        # class_nums = [1256, 281, 50, 208, 402, 68, 345]
    elif args.dataset == 'FINANCE':
        class_nums = [7313,1855,728,1116,862]

    total_samples = sum(class_nums)
    '''
    alpha 的计算公式是基于 total_samples / count，这通常在机器学习中与样本权重相关联，特别是当我们要处理类别不平衡问题时。
    在这种情况下，每个类别的 alpha 值代表了相对于整个数据集中样本总数而言，该类别样本量的倒数比例。
    换句话说，它试图为样本较少的类别分配更高的权重，从而在训练模型时平衡不同类别的影响。
    '''
    alpha = torch.tensor([total_samples / count for count in class_nums])
    alpha /= alpha.sum()  # 归一化
    loss_focalloss_func = FocalLoss(class_num=args.class_num, alpha=alpha, gamma=args.gamma, size_average=True)

    # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
    model.train() if mode == 'train' else model.eval()
    loss_sum = 0
    preds = []
    labels = []
    epoch_lr = []  # 记录每轮学习率
    '''
    这段代码的作用是在一个代码块中动态地开启或关闭PyTorch中的自动梯度计算。这是通过torch.set_grad_enabled函数实现的，它可以在运行时控制是否需要追踪计算图以计算梯度。
    '''
    with torch.set_grad_enabled(mode == 'train'):
        for step, batch in enumerate(tqdm(dataloader, disable=True)):
            sentence_vectors, edge_idx_other, label, edge_idx_speaker, edge_idx_longdis = batch
            output = model(sentence_vectors.squeeze(0), edge_idx_other.squeeze(0), edge_idx_speaker.squeeze(0), edge_idx_longdis.squeeze(0))
            label = label.squeeze(0)
            loss_class = loss_cross_entropy_func(output, label)
            if args.gradient_accumulation_step > 1:
                # loss /= label.size(0)
                loss_class /= args.gradient_accumulation_step

            loss_sum += loss_class.item()
            if mode == 'train':
                loss_class.backward()
                if (step + 1) % args.gradient_accumulation_step == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    current_lr = scheduler.get_last_lr()[0]
                    epoch_lr.append(current_lr)

            preds.extend(torch.argmax(output.detach(), 1).cpu().numpy().tolist())
            labels.extend(label.cpu().numpy().tolist())
        # print("@@@@@@@@@@@",preds, labels)
    return loss_sum / len(dataloader), preds, labels, epoch_lr

random_seed = random.randint(1, 1000)
set_random_seeds(random_seed)
random_int = random.randint(1, 1000) # 模型名字不至于重复
print("随机种子:", random_seed, '随机数', random_int)

##Create Output Directory if it doesn't exists
# config['model_internal_dir'] = datetime.datetime.now().strftime('%m%d%Y_%H%M%S')
# if not os.path.exists(config['model_output_dir']):
#     os.mkdir(config['model_output_dir'])

# os.mkdir(Path(config['model_output_dir']) / config['model_internal_dir'])
#
# print("Model Dir: ", Path(config['model_output_dir']) / config['model_internal_dir'])

# Create All Data Loaders
print("Preparing Datasets and DataLoaders")
if args.dataset == 'MELD':
    train_dataset = MELDDataset(train = True)
    test_dataset = MELDDataset(train = False)
elif args.dataset == 'IEMOCAP':
    train_dataset = IEMOCAPDataset(train = True)
    test_dataset = IEMOCAPDataset(train = False)
elif args.dataset == 'FINANCE':
    train_dataset = FinanceDataset(train = True)
    test_dataset = FinanceDataset(train = False)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)
print("Datasets and DataLoaders Done! :D")

args.num_train_optimization_steps = (int(len(train_dataset) / args.batch_size / args.gradient_accumulation_step) * args.num_epochs)

print("Creating Model")
model, optimizer, scheduler = fetch_model_optim_sched(device,args)
print("Model Created! :D")
from sklearn.metrics import classification_report, f1_score
import warnings
warnings.filterwarnings("ignore")

print("Starting Training")
print("随机种子:", random_seed, '随机数', random_int)
print('参数：', args)
if args.dataset == 'IEMOCAP':
    target_names = ['hap', 'sad', 'neu', 'ang', 'exc', 'fru']
elif args.dataset =='MELD':
    target_names = ['neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger']
elif args.dataset == 'FINANCE':
    target_names =['中性','积极','困惑','伤心','愤怒']



best_val_loss = None

for epoch in range(args.num_epochs):
    print(f"Epoch: {epoch}")
    # 打印学习率
    # 获取每个参数组的学习率
    learning_rates = []
    for param_group in optimizer.param_groups:
        learning_rates.append(param_group['lr'])

    # 打印学习率
    for i, lr in enumerate(learning_rates):
        print("Learning rate for parameter group {}: {}".format(i, lr))

    tloss, train_epoch_preds, train_epoch_labels, epoch_lr = loop(train_loader, model, optimizer, scheduler, device, args, mode='train')
    testloss, test_epoch_preds,test_epoch_labels, epoch_lr= loop(test_loader, model, optimizer, scheduler, device, args, mode='val')

    testf1 = f1_score(test_epoch_labels,test_epoch_preds ,average='weighted')
    trainf1 = f1_score(train_epoch_labels,train_epoch_preds ,average='weighted')
    # print('学习率',epoch_lr)
    if best_val_loss is None:
        best_val_loss = testloss
    elif best_val_loss > testloss:
        best_val_loss = testloss
        print('testloss loss 下降了↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓')

        # 构造模型名称
        model_name = f"{args.dataset}_seed_{random_seed}_int_{random_int}_head_{args.num_heads}"

        # 拼接路径
        PATH = os.path.join(args.model_output_dir, f"{model_name}.ckt")

        print('正在保存模型',PATH)
        torch.save(model.state_dict(), PATH)
        print('--------------------train--------------')

        print(classification_report(train_epoch_labels,train_epoch_preds,digits=4, target_names=target_names))
        print('--------------------test--------------')
        print(classification_report(test_epoch_labels,test_epoch_preds,digits=4, target_names=target_names))

    print("Train Loss: {:.4f}, Train F1: {:.4f} | | Test Loss: {:.4f}, Test F1: {:.4f}".format(tloss, trainf1, testloss, testf1))


print("Finished Training! :D")
# wandb.finish(0)
