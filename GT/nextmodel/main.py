import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', default=1e-3, type=float, help='Learning Rate')  # 定义学习率
parser.add_argument('--warmup_proportion', default=0.1, type=float,
                    help='Set Warmup Proportion')  # 定义学习率预热比例 如果设置warmup_proportion为0.1，意味着在训练的前10%的步骤中，学习率会从一个非常小的值（可能是接近于0）逐渐线性增加到其设定的初始值。在预热阶段结束后，学习率保持在初始值，或者按照其他的调度策略（如线性衰减、余弦退火等）进行调整。
parser.add_argument('--num_epochs', default=20, type=int, help='Set Number of Epochs')
parser.add_argument('--batch_size', default=1, type=int, help='Set Batch Size')
parser.add_argument('--mode', default='train', type=str, choices=['train', 'test'], help='Train Or Test Mode')
parser.add_argument('--gradient_accumulation_step', default=8,
                    type=int)  # 8               #定义了一个整型参数--gradient_accumulation_step，默认值是8，用于控制梯度累积的步数，这对于在有限的GPU内存下使用更大的有效批量是有用的。
parser.add_argument('--model_output_dir', default='output/meld/', choices=['output/meld/', 'output/iemocap/'], type=str)
# lineConGraph 定义 argparse 参数
parser.add_argument('--in_features', type=int, default=128, help='输入文本特征的维度')
parser.add_argument('--hidden_features1', type=int, default=64, help='隐藏层特征的维度')
parser.add_argument('--hidden_features2', type=int, default=32, help='隐藏层特征的维度')
parser.add_argument('--out_features', type=int, default=7,
                    help='输出特征的维度')  # IEMOCAP 6['hap', 'sad', 'neu', 'ang', 'exc', 'fru'] MELD 7['neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger']
parser.add_argument('--num_heads', type=int, default=8, help='GATConv 中的注意力头数')  # IEMOCAP 1头好 MELD 4头好
parser.add_argument('--momentum', type=float, default=0.9, help='SGD优化器(随机梯度下降)的动量参数')
parser.add_argument('--dataset', type=str, default='MELD', choices=['IEMOCAP', 'MELD'],
                    help='Dataset to use (IEMOCAP or MELD)')
parser.add_argument('--gamma', type=float, default=3, help='focalloss的gamma')
parser.add_argument('--n', type=int, default=8, help='loss的n')
parser.add_argument('--beta', type=float, default=5e-8, help='beta')
parser.add_argument('--theta', type=float, default=0.001, help='theta')
parser.add_argument('--audio_features', type=int, default=300, choices=[300, 100], help='输入音频特征的维度')
parser.add_argument('--visual_features', type=int, default=342, choices=[342, 256], help='输入视频特征的维度')

args = parser.parse_args([])
if args.dataset == 'IEMOCAP':
    args.out_features = 6
    args.class_num = 6
    args.model_output_dir = 'output/iemocap/'
    args.audio_features = 100
    args.visual_features = 256
    args.num_speakers = 2
elif args.dataset == 'MELD':
    args.out_features = 7
    args.class_num = 7
    args.model_output_dir = 'output/meld/'
    args.audio_features = 300
    args.visual_features = 342
    args.num_speakers = 9
print(args)
