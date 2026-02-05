""" Training VAE """
import argparse
from os.path import join, exists
from os import mkdir

import torch
import torch.utils.data
from torch import optim
from torch.nn import functional as F
from torchvision import transforms#数据预处理（裁剪、翻转、归一化）
from torchvision.utils import save_image#保存生成的样本图片

from models.vae import VAE

from utils.misc import save_checkpoint
from utils.misc import LSIZE, RED_SIZE
## WARNING : THIS SHOULD BE REPLACE WITH PYTORCH 0.5
from utils.learning import EarlyStopping
from utils.learning import ReduceLROnPlateau
from data.loaders import _RolloutDataset
"""
整体作用：
训练一个用于深度图的 VAE（变分自编码器），输出权重文件 best.tar。
该权重会被后续的 LSTM 训练 和 PPO 训练中的特征提取器初始化 使用。

调用与被调用关系：
调用方式：作为脚本直接运行（README 提到 python trainvae.py）。
注意：这个文件没有 if __name__ == "__main__"，只要被 import 就会立刻开始训练。

输入/输出
输入数据：saved/lstm_dataset（（由 collect_data.py 生成））。
输出权重：exp_vae/vae/best.tar（最佳权重）+ checkpoint.tar（检查点）+ 样本图片
样本可视化：sample_*.png（每个 epoch 生成）。

"""
#1、参数与设备设置
parser = argparse.ArgumentParser(description='VAE Trainer')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',#单次训练的样本数
                    help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',#	最多训练轮数（主循环 for epoch in range(1, args.epochs + 1) 使用
                    help='number of epochs to train (default: 1000)')
parser.add_argument('--logdir', type=str,  default='exp_vae', help='Directory where results are logged')
parser.add_argument('--noreload', action='store_true',#决定是否加载已有 best.tar
                    help='Best model is not reloaded if specified')
parser.add_argument('--nosamples', action='store_true',#决定是否生成并保存样本图像
                    help='Does not save samples during training if specified')


args = parser.parse_args()
cuda = torch.cuda.is_available()


torch.manual_seed(123)
# Fix numeric divergence due to bug in Cudnn
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if cuda else "cpu")

#2、数据处理与加载
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((RED_SIZE, RED_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((RED_SIZE, RED_SIZE)),
    transforms.ToTensor(),
])

dataset_train = _RolloutDataset('saved/lstm_dataset',
                                          transform_train, train=True)
dataset_test = _RolloutDataset('saved/lstm_dataset',
                                         transform_test, train=False)
train_loader = torch.utils.data.DataLoader(
    dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=1)
test_loader = torch.utils.data.DataLoader(
    dataset_test, batch_size=args.batch_size, shuffle=True, num_workers=1)

#3、模型、优化器
model = VAE(1, LSIZE).to(device)# ① 模型初始化
optimizer = optim.Adam(model.parameters())# ② 优化器：Adam 优化器，学习率默认 0.001
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)# ③ 学习率调度器：若测试损失停止下降，则降低学习率
earlystopping = EarlyStopping('min', patience=50)# ④ 早停策略：若 50 个 epoch 后测试损失无改进，停止训练

# 4、关键函数 Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logsigma):
    """ VAE loss function """
    BCE = F.mse_loss(recon_x, x, reduction='sum')
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # #KL 散度损失（Kullback-Leibler Divergence Loss）
    # 约束潜在空间分布接近标准正态分布 N(0,1)
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + 2 * logsigma - mu.pow(2) - (2 * logsigma).exp())
    return BCE + KLD# 总损失 = 重构 + KL 散度


def train(epoch):
    """ One training epoch """
    model.train()# 开启训练模式（影响 Dropout、BatchNorm）
    dataset_train.load_next_buffer()# 从磁盘加载数据到内存
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)# 数据转移到 GPU/CPU
        # print(data.shape)
        # ① 清零梯度
        optimizer.zero_grad()
        # ② 前向传播
        # recon_batch: 重构图像、mu: 潜在空间均值、
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        # ④ 反向传播
        loss.backward()
        train_loss += loss.item()
        # ⑤ 更新权重
        optimizer.step()
        # 每 20 个 batch 打印进度
        if batch_idx % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))


def test():
    """ One test epoch """
    model.eval()# 开启评估模式（关闭 Dropout 等）
    dataset_test.load_next_buffer()
    test_loss = 0
    with torch.no_grad():# 禁用梯度计算（节省内存、加速推理）
        for data in test_loader:
            data = data.to(device)
            # 前向传播
            recon_batch, mu, logvar = model(data)
            # 计算损失（无梯度反向传播）
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
    # 返回平均测试损失
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss
#5、主训练循环
# check vae dir exists, if not, create it
# 创建保存目录
vae_dir = join(args.logdir, 'vae')
if not exists(vae_dir):
    mkdir(vae_dir)
    mkdir(join(vae_dir, 'samples'))
# 重新加载历史权重（如果存在且 --noreload 未设置）
reload_file = join(vae_dir, 'best.tar')
if not args.noreload and exists(reload_file):
    state = torch.load(reload_file)
    print("Reloading model at epoch {}"
          ", with test error {}".format(
              state['epoch'],
              state['precision']))
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    earlystopping.load_state_dict(state['earlystopping'])


cur_best = None
# ① 每个 epoch 循环
for epoch in range(1, args.epochs + 1):
    train(epoch)
    test_loss = test()
    # ② 学习率调度
    scheduler.step(test_loss)
    # ③ 早停检查
    earlystopping.step(test_loss)

    # checkpointing
    best_filename = join(vae_dir, 'best.tar')
    filename = join(vae_dir, 'checkpoint.tar')
    # ④ 检查点保存
    is_best = not cur_best or test_loss < cur_best
    if is_best:
        cur_best = test_loss

    save_checkpoint({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'precision': test_loss,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'earlystopping': earlystopping.state_dict()
    }, is_best, filename, best_filename)


    # ⑤ 样本生成与保存
    if not args.nosamples:
        with torch.no_grad():
            # 从标准正态分布采样向量
            sample = torch.randn(RED_SIZE, LSIZE).to(device)
            # 通过解码器生成图像
            sample = model.decoder(sample).cpu()
            # 保存生成的图像
            save_image(sample.view(256, 1, RED_SIZE, RED_SIZE),
                       join(vae_dir, 'samples/sample_' + str(epoch) + '.png'))
    # ⑥ 早停触发
    if earlystopping.stop:
        print("End of Training because of early stopping at epoch {}".format(epoch))
        break
