import torch
import torch.nn as nn
import numpy as np
import argparse
from data_loader import get_loader, to_categorical


# 条件实例归一化网络
class ConditionalInstanceNormalisation(nn.Module):
    """CIN块"""
    def __init__(self, dim_in, style_num):
        super(ConditionalInstanceNormalisation, self).__init__()
        # 定义cpu或者gpu运行
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dim_in = dim_in
        # 风格数量
        self.style_num = style_num
        # 两个学习参数
        self.gamma = nn.Linear(style_num, dim_in)
        self.beta = nn.Linear(style_num, dim_in)

    def forward(self, x, c):
        # 根据行计算平均数并保持维数
        u = torch.mean(x, dim=2, keepdim=True)
        # 计算标准差
        var = torch.mean((x - u) * (x - u), dim=2, keepdim=True)
        std = torch.sqrt(var + 1e-8)
        # width = x.shape[2]
        gamma = self.gamma(c.to(self.device))
        gamma = gamma.view(-1, self.dim_in, 1)
        beta = self.beta(c.to(self.device))
        beta = beta.view(-1, self.dim_in, 1)
        # 按照论文进行运算
        h = (x - u) / std
        h = h * gamma + beta

        return h


# 残余块
class ResidualBlock(nn.Module):
    """具有实例归一化的剩余块"""
    def __init__(self, dim_in, dim_out, style_num):
        super(ResidualBlock, self).__init__()
        # 定义一个一维卷积
        self.conv_1 = nn.Conv1d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False)
        # 定义一个CIN
        self.cin_1 = ConditionalInstanceNormalisation(dim_out, style_num)
        # 定义一个门控线性激活函数
        self.relu_1 = nn.GLU(dim=1)

    def forward(self, x, c):
        x_ = self.conv_1(x)
        x_ = self.cin_1(x_, c)
        x_ = self.relu_1(x_)

        return x_


class Generator(nn.Module):
    """生成器网络"""
    def __init__(self, num_speakers=4):
        super(Generator, self).__init__()
        # 下采样层
        self.down_sample_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(3, 9), padding=(1, 4), bias=False),
            nn.GLU(dim=1)
        )
        self.down_sample_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(4, 8), stride=(2, 2), padding=(1, 3), bias=False),
            nn.InstanceNorm2d(num_features=256, affine=True, track_running_stats=True),
            nn.GLU(dim=1)
        )
        self.down_sample_3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(4, 8), stride=(2, 2), padding=(1, 3), bias=False),
            nn.InstanceNorm2d(num_features=512, affine=True, track_running_stats=True),
            nn.GLU(dim=1)
        )

        # 下转换层
        self.down_conversion = nn.Sequential(
            nn.Conv1d(in_channels=2304,
                      out_channels=256,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.InstanceNorm1d(num_features=256, affine=True)
        )

        # 瓶颈层，用于特色转换
        self.residual_1 = ResidualBlock(dim_in=256, dim_out=512, style_num=num_speakers)
        self.residual_2 = ResidualBlock(dim_in=256, dim_out=512, style_num=num_speakers)
        self.residual_3 = ResidualBlock(dim_in=256, dim_out=512, style_num=num_speakers)
        self.residual_4 = ResidualBlock(dim_in=256, dim_out=512, style_num=num_speakers)
        self.residual_5 = ResidualBlock(dim_in=256, dim_out=512, style_num=num_speakers)
        self.residual_6 = ResidualBlock(dim_in=256, dim_out=512, style_num=num_speakers)
        self.residual_7 = ResidualBlock(dim_in=256, dim_out=512, style_num=num_speakers)
        self.residual_8 = ResidualBlock(dim_in=256, dim_out=512, style_num=num_speakers)
        self.residual_9 = ResidualBlock(dim_in=256, dim_out=512, style_num=num_speakers)

        # 上转换层
        self.up_conversion = nn.Conv1d(in_channels=256,
                                       out_channels=2304,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0,
                                       bias=False)

        # 上采样层
        self.up_sample_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(num_features=256, affine=True, track_running_stats=True),
            nn.GLU(dim=1)
        )
        self.up_sample_2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(num_features=128, affine=True, track_running_stats=True),
            nn.GLU(dim=1)
        )

        # 输出
        self.out = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=7, stride=1, padding=3, bias=False)

    def forward(self, x, c):
        # 获取输入数据的宽度
        width_size = x.size(3)
        # 进行三次下采样层
        x = self.down_sample_1(x)
        x = self.down_sample_2(x)
        x = self.down_sample_3(x)
        # 执行view操作之后，不会开辟新的内存空间来存放处理之后的数据，实际上新数据与原始数据共享同一块内存。
        # 而在调用contiguous()之后，PyTorch会开辟一块新的内存空间存放变换之后的数据，并会真正改变Tensor的内容，按照变换之后的顺序存放数据。
        # 对x数据将宽度减少到原来的1/4
        x = x.contiguous().view(-1, 2304, width_size // 4)
        # 下转换层
        x = self.down_conversion(x)
        # 9个瓶颈层（用来风格转换）
        x = self.residual_1(x, c)
        x = self.residual_2(x, c)
        x = self.residual_3(x, c)
        x = self.residual_4(x, c)
        x = self.residual_5(x, c)
        x = self.residual_6(x, c)
        x = self.residual_7(x, c)
        x = self.residual_8(x, c)
        x = self.residual_9(x, c)
        # 上转换层
        x = self.up_conversion(x)
        x = x.view(-1, 256, 9, width_size // 4)
        # 两个上采样层
        x = self.up_sample_1(x)
        x = self.up_sample_2(x)
        x = self.out(x)

        return x


class Discriminator(nn.Module):
    """判别器网络"""
    def __init__(self, num_speakers=10):
        super(Discriminator, self).__init__()
        self.num_speakers = num_speakers
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # 初始化层
        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.GLU(dim=1)
        )
        # 下采样层
        self.down_sample_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.InstanceNorm2d(num_features=256, affine=True, track_running_stats=True),
            nn.GLU(dim=1)
        )
        self.down_sample_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.InstanceNorm2d(num_features=512, affine=True, track_running_stats=True),
            nn.GLU(dim=1)
        )
        self.down_sample_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.InstanceNorm2d(num_features=1024, affine=True, track_running_stats=True),
            nn.GLU(dim=1)
        )
        self.down_sample_4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2), bias=False),
            nn.InstanceNorm2d(num_features=1024, affine=True, track_running_stats=True),
            nn.GLU(dim=1)
        )
        # 全连接层
        self.fully_connected = nn.Linear(in_features=512, out_features=1)
        # 映射层.
        self.projection = nn.Linear(self.num_speakers, 512)

    def forward(self, x, c, c_):
        # c_onehot = torch.cat((c, c_), dim=1).to(self.device)
        c_onehot = c_
        # 初始化
        x = self.conv_layer_1(x)
        # 四个下采样层
        x = self.down_sample_1(x)
        x = self.down_sample_2(x)
        x = self.down_sample_3(x)
        x_ = self.down_sample_4(x)
        # 从第3维和4维计算总和
        h = torch.sum(x_, dim=(2, 3))
        # 全连接层
        x = self.fully_connected(h)
        # 将c的one-hot格式加入映射
        p = self.projection(c_onehot)
        x += torch.sum(p * h, dim=1, keepdim=True)

        return x


# 只是为了测试结构的形状
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='检查G和D的结构')
    train_dir_default = '../data/VCTK-Data/mc/train'
    speaker_default = 'p229'
    # 数据配置
    parser.add_argument('--train_dir', type=str, default=train_dir_default, help='Train dir path')
    parser.add_argument('--speakers', type=str, nargs='+', required=True, help='Speaker dir names')
    num_speakers = 4
    argv = parser.parse_args()
    train_dir = argv.train_dir
    speakers_using = argv.speakers
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载模型
    generator = Generator(num_speakers=num_speakers).to(device)
    discriminator = Discriminator(num_speakers=num_speakers).to(device)
    # 加载数据
    train_loader = get_loader(speakers_using, train_dir, 8, 'train', num_workers=1)
    data_iter = iter(train_loader)

    mc_real, spk_label_org, spk_c_org = next(data_iter)
    mc_real.unsqueeze_(1)  # (B, D, T) -> (B, 1, D, T) for conv2d

    spk_c = np.random.randint(0, num_speakers, size=mc_real.size(0))
    spk_c_cat = to_categorical(spk_c, num_speakers)
    spk_label_trg = torch.LongTensor(spk_c)
    spk_c_trg = torch.FloatTensor(spk_c_cat)

    mc_real = mc_real.to(device)              # Input mc.
    spk_label_org = spk_label_org.to(device)  # Original spk labels.
    spk_c_org = spk_c_org.to(device)          # Original spk acc conditioning.
    spk_label_trg = spk_label_trg.to(device)  # Target spk labels for classification loss for G.
    spk_c_trg = spk_c_trg.to(device)          # Target spk conditioning.

    print('------------------------')
    print('测试判别器')
    print('-------------------------')
    print(f'输入的形状：{mc_real.shape}')
    dis_real = discriminator(mc_real, spk_c_org, spk_c_trg)
    print(f'输出的形状：{dis_real.shape}')
    print('------------------------')

    print('测试生成器')
    print('-------------------------')
    print(f'输入的形状：{mc_real.shape}')
    mc_fake = generator(mc_real, spk_c_trg)
    print(f'输出的形状：{mc_fake.shape}')
    print('------------------------')
