import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
import random
import warnings
import numpy as np
import torch.nn as nn
from main import args
from tqdm.auto import tqdm
from logo import logo_model
from torch.optim import Adam
from nextmodel import Graph_Construction
import torch.nn.functional as F
from torch.autograd import Variable
from meldDataset import MELDDataset
from torch.utils.data import DataLoader
from iemocapDataset import IEMOCAPDataset
from sklearn.metrics import classification_report, f1_score
from transformers import AdamW, get_cosine_schedule_with_warmup
from sklearn.metrics import confusion_matrix

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FocalLoss(nn.Module):
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
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, dim=1)

        class_mask = inputs.data.new(N, C).fill_(0)  # 创建和inputs.data类型，形状为(N,C)全0的tensor
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)  # 在PyTorch中对张量class_mask进行操作，用于创建one-hot编码向量或矩阵

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()  # probs.对数值
        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


def set_random_seeds(seed):
    print("Seed: {}".format(seed))

    torch.backends.cudnn.benchmark = False  # 将它设置为False禁用了这个特性，确保每次运行都使用相同的卷积算法。
    # 这一行完全禁用了cuDNN的使用。cuDNN是NVIDIA提供的深度神经网络库，可以加速在GPU上的深度学习计算。禁用它意味着所有的操作都不会使用cuDNN，而是使用PyTorch的默认实现。这同样有助于确保结果的一致性，但可能会降低性能，尤其是对于大规模的卷积操作。
    torch.backends.cudnn.enabled = False
    # 将cudnn.deterministic设置为True强制cuDNN使用确定性的算法，即使它们可能不是最快的。这有助于确保在相同输入下的多次运行产生相同的结果。
    torch.backends.cudnn.deterministic = True

    random.seed(seed)  # 用于设置随机数生成器的种子
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fetch_model_optim_sched(device, args):
    model = Graph_Construction(args.audio_features,args.visual_features,args.in_features,args.num_speakers,args)
    model.to(device)

    # optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    optimizer = Adam(model.parameters(), lr=args.learning_rate,weight_decay=1e-5)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_proportion * args.num_train_optimization_steps,
        num_training_steps=args.num_train_optimization_steps,
    )

    return model, optimizer, scheduler


def loop(dataloader, model, optimizer, scheduler, device, args, mode='train'):
    loss_cross_entropy_func = F.cross_entropy

    if args.dataset == 'IEMOCAP':
        class_nums = [504, 839, 1324, 933, 742, 1468]
        # class_nums = [135, 229, 379, 168, 295, 358]
    else:
        class_nums = [5180, 1355, 308, 794, 1906, 293, 1262]
        # class_nums = [1256, 281, 50, 208, 402, 68, 345]
    total_samples = sum(class_nums)
    '''
    alpha 的计算公式是基于 total_samples / count，这通常在机器学习中与样本权重相关联，特别是当我们要处理类别不平衡问题时。
    在这种情况下，每个类别的 alpha 值代表了相对于整个数据集中样本总数而言，该类别样本量的倒数比例。
    换句话说，它试图为样本较少的类别分配更高的权重，从而在训练模型时平衡不同类别的影响。
    '''
    alpha = torch.tensor([total_samples / count for count in class_nums])
    alpha /= alpha.sum()  # 归一化
    # loss_focalloss_func = FocalLoss(class_num=args.class_num, alpha=alpha, gamma=args.gamma, size_average=True)

    model.train() if mode == 'train' else model.eval()
    loss_sum = 0
    preds = []
    labels = []
    epoch_lr = []  # 记录每轮学习率
    '''
    这段代码的作用是在一个代码块中动态地开启或关闭PyTorch中的自动梯度计算。这是通过torch.set_grad_enabled函数实现的，它可以在运行时控制是否需要追踪计算图以计算梯度。
    '''
    with torch.set_grad_enabled(mode == 'train'):
        for step, batch in enumerate(tqdm(dataloader, desc=f"{mode.capitalize()} Progress", disable=False)):
            videoAudio,videoVisual,sentence_vectors, speaker_vectors,label = batch
            output = model(videoAudio.squeeze(0), videoVisual.squeeze(0).to(torch.float32), sentence_vectors.squeeze(0),speaker_vectors)
            label = label.squeeze(0)
            loss_class = loss_cross_entropy_func(output,label)
            # loss_class = loss_focalloss_func(output, label)
            if args.gradient_accumulation_step > 1:
                loss_class /= args.gradient_accumulation_step

            if mode == 'train':
                loss_class.backward()
                if (step + 1) % args.gradient_accumulation_step == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    current_lr = scheduler.get_last_lr()[0]
                    epoch_lr.append(current_lr)
            loss_sum += loss_class.item()
            preds.extend(torch.argmax(output.detach(), 1).cpu().numpy().tolist())
            labels.extend(label.cpu().numpy().tolist())
    return loss_sum / len(dataloader), preds, labels, epoch_lr


def prepare_datasets(args):
    if args.dataset == 'MELD':
        train_dataset = MELDDataset(train=True)
        test_dataset = MELDDataset(train=False)
    else:
        train_dataset = IEMOCAPDataset(train=True)
        test_dataset = IEMOCAPDataset(train=False)

    return train_dataset, test_dataset


def create_data_loaders(train_dataset, test_dataset, batch_size):
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader


def create_model(args, device):
    model, optimizer, scheduler = fetch_model_optim_sched(device, args)
    return model, optimizer, scheduler


def train_model(model, train_loader, val_loader, optimizer, scheduler, device, args):
    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = -1

    for epoch in range(args.num_epochs):
        print(f"Epoch: {epoch}")
        learning_rates = [param_group['lr'] for param_group in optimizer.param_groups]
        for i, lr in enumerate(learning_rates):
            print("Learning rate for parameter group {}: {}".format(i, lr))

        tloss, train_epoch_preds, train_epoch_labels, epoch_lr = loop(train_loader, model, optimizer, scheduler, device,
                                                                      args, mode='train')

        val_loss, val_epoch_preds, val_epoch_labels, _ = loop(val_loader, model, optimizer, scheduler, device, args,
                                                              mode='val')
        val_f1 = f1_score(val_epoch_labels, val_epoch_preds, average='weighted')
        print()
        print(f'损失值为:{val_loss},验证集F1为:{val_f1}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            best_epoch = epoch
            print(logo_model())
            print('Validation Classification Report:')
            print(classification_report(val_epoch_labels, val_epoch_preds, target_names=args.target_names))

    return best_model_state, best_epoch


def evaluate_model(model, test_loader, device, args):
    model.eval()
    test_loss, test_epoch_preds, test_epoch_labels, _ = loop(test_loader, model, None, None, device, args, mode='val')
    test_f1 = f1_score(test_epoch_labels, test_epoch_preds, average='weighted')

    print('Test Classification Report:')
    print(classification_report(test_epoch_labels, test_epoch_preds, target_names=args.target_names))
    print("Test Loss: {:.4f}, Test F1: {:.4f}".format(test_loss, test_f1))


def main(args):
    random_seed = random.randint(1, 1000)
    random.seed(random_seed)
    print("随机种子:", random_seed)
    print("Preparing Datasets and DataLoaders")

    train_dataset, test_dataset = prepare_datasets(args)
    train_loader, val_loader, test_loader = create_data_loaders(train_dataset, test_dataset, args.batch_size)

    print("Datasets and DataLoaders Done! :D")
    args.num_train_optimization_steps = (
            int(len(train_dataset) / args.batch_size / args.gradient_accumulation_step) * args.num_epochs)

    print("Creating Model")
    model, optimizer, scheduler = create_model(args, device)
    print("Model Created! :D")

    warnings.filterwarnings("ignore")
    print("Starting Training")

    args.target_names = ['hap', 'sad', 'neu', 'ang', 'exc', 'fru'] if args.dataset == 'IEMOCAP' else ['neutral',
                                                                                                      'surprise',
                                                                                                      'fear', 'sadness',
                                                                                                      'joy', 'disgust',
                                                                                                      'anger']

    best_model_state, best_epoch = train_model(model, train_loader, val_loader, optimizer, scheduler, device, args)

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f'最佳训练周期为{best_epoch}')
        evaluate_model(model, test_loader, device, args)

    print("Finished Training! :D")


if __name__ == '__main__':
    main(args)
