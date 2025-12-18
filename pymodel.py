import numpy as np
import torch
import torch.nn as nn
import random
from torchsummary import summary
import math




class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b,c,h,w = x.size()
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        y = torch.ones(size=(b,c,h,w),dtype=torch.float32).cuda()
        z = torch.zeros(size=(b,c,h,w),dtype=torch.float32).cuda()
        beta = 0.2
        # change the value of beta to acquire best results
        out = torch.where(out.data>=beta,out,z)
        # print(out.grad)

        return out



class MixStyle(nn.Module):
    """MixStyle.
    Reference:
      Zhou et al. Domain Generalization with MixStyle. ICLR 2021.
    """

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, mix='random'):
        """
        Args:
          p (float): probability of using MixStyle.
          alpha (float): parameter of the Beta distribution.
          eps (float): scaling parameter to avoid numerical issues.
          mix (str): how to mix.
        """
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha
        self.mix = mix
        self._activated = True

    def __repr__(self):
        return f'MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})'

    def set_activation_status(self, status=True):
        self._activated = status

    def update_mix_method(self, mix='crossdomain'):
        self.mix = mix

    def forward(self, x):
        if not self.training or not self._activated:
            return x  #64*64*5*5

        P = random.random()
        if P > self.p:
            return x

        B = x.size(0) #64

        mu = x.mean(dim=[2, 3], keepdim=True) #64*64*1*1
        var = x.var(dim=[2, 3], keepdim=True) #64*64*1*1 方差
        sig = (var + self.eps).sqrt() #64*64*1*1 标准差
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x-mu) / sig #64*64*5*5 归一化

        lmda = self.beta.sample((B, 1, 1, 1)) #64*1*1*1 从Beta分布中采样一个参数lmda，用于控制两个样本之间的混合程度。
        lmda = lmda.to(x.device)

        if self.mix == 'random':
            # random shuffle
            perm = torch.randperm(B) #随机打乱索引 0-64

        elif self.mix == 'crossdomain':
            # split into two halves and swap the order
            perm = torch.arange(B - 1, -1, -1) # inverse index
            size1 = B // 2
            size2 = B - size1
            perm_b = perm[:size1]
            perm_a = perm[size1:]
            perm_b = perm_b[torch.randperm(size1)]
            perm_a = perm_a[torch.randperm(size2)]
            perm = torch.cat([perm_b, perm_a], 0)

        else:
            raise NotImplementedError

        mu2, sig2 = mu[perm], sig[perm] #使用 perm 索引获取其他样本的统计量 mu2 和 sig2。
        mu_mix = mu*lmda + mu2 * (1-lmda) #64*64*1*1
        sig_mix = sig*lmda + sig2 * (1-lmda)

        return x_normed*sig_mix + mu_mix #64*64*5*5


class HyperGroupMix(nn.Module):
    """
    Reference:
      Zhou et al. Domain Generalization with MixStyle. ICLR 2021.
    """

    def __init__(self, groups=[[0,16],[16,32],[32,64]],p=0.5, alpha=0.1, eps=1e-6, mix='random'):

        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha
        self.mix = mix
        self._activated = True
        self.groups = groups

    def __repr__(self):
        return f'MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})'

    def set_activation_status(self, status=True):
        self._activated = status

    def update_mix_method(self, mix='crossdomain'):
        self.mix = mix

    def forward(self, x):
        if not self.training or not self._activated:
            return x  #64*64*5*5

        P = random.random()
        if P > self.p:
            return x

        #B = x.size(0) #64

        out = []
        for start, end in self.groups:
            x_g = x[:, start:end, :, :]
            B, C, H, W = x_g.shape
            mu = x_g.mean(dim=[2, 3], keepdim=True) #64*64*1*1
            var = x_g.var(dim=[2, 3], keepdim=True) #64*64*1*1 方差
            sig = (var + self.eps).sqrt().detach()#64*64*1*1 标准差
            mu, sig = mu.detach(), sig.detach()
            x_normed = (x_g - mu) / sig  # 64*64*5*5 归一化
            del  var, sig

            median_h = x_g.median(dim=2, keepdim=True)[0]
            median = median_h.median(dim=3, keepdim=True)[0].detach()

            # 计算Gram矩阵并添加扰动
            flat_x = x_g .view(B, C, -1) #64*64*25
            gram = torch.bmm(flat_x, flat_x.transpose(1, 2)) / (H * W + self.eps) #64*64*64 B,C,C
            gram = gram.detach()
            gram = gram + 1e-3 * torch.eye(C, device=x_g.device).unsqueeze(0) + 1e-6 *  torch.eye(C, device=x_g.device).unsqueeze(0)

            lmda = self.beta.sample((B, 1, 1, 1)) #64*1*1*1 从Beta分布中采样一个参数lmda，用于控制两个样本之间的混合程度。
            lmda = lmda.to(x_g.device)

            if self.mix == 'random':
                # random shuffle
                perm = torch.randperm(B) #随机打乱索引 0-64

            elif self.mix == 'crossdomain':
                # split into two halves and swap the order
                perm = torch.arange(B - 1, -1, -1) # inverse index
                size1 = B // 2
                size2 = B - size1
                perm_b = perm[:size1]
                perm_a = perm[size1:]
                perm_b = perm_b[torch.randperm(size1)]
                perm_a = perm_a[torch.randperm(size2)]
                perm = torch.cat([perm_b, perm_a], 0)

            else:
                raise NotImplementedError

            median_mix = median * lmda + median[perm] * (1 - lmda)# (B, C, 1, 1)

            # 对 Gram 矩阵进行特征分解
            eigvals, eigvecs = torch.linalg.eigh(gram)  # (64, 64), (64, 64, 64)
            eigvals = eigvals + self.eps  # 避免除零

            # 计算白化矩阵: W = V * diag(1/sqrt(λ)) * V^T
            whitening = eigvecs @ torch.diag_embed(1.0 / torch.sqrt(eigvals)) @ eigvecs.transpose(1, 2)  # (64, 64, 64)

            # 应用白化变换
            x_whitened = torch.bmm(whitening, x_g.view(B, C, -1) - mu.view(B, C, -1))  #64*64*25 (B, C, H*W)
            x_whitened = x_whitened.view(B, C, H, W)  #64*64*5*5 (B, C, H, W)

            gram_mix = x_whitened * lmda + x_whitened[perm] * (1 - lmda)  # 64*64*5*5
            del x_whitened, whitening, eigvals, eigvecs, flat_x, gram
            out.append(x_normed*gram_mix + median_mix)

        return torch.cat(out,dim=1) #64*64*5*5



class CMFM(nn.Module):
    def __init__(self,FM):
        super().__init__()
        self.conv1x1 = nn.Conv2d(FM*8, FM*4, kernel_size=1)
        #self.norm = nn.LayerNorm(FM * 4)
        self.BN = nn.BatchNorm2d(FM * 4)
        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.3)

    def process_tensor(self, x):
        b, c, h, w = x.size()
        x_unfold = x.unfold(2, 2, 2).unfold(3, 2, 2)
        x_processed = x_unfold.contiguous().view(b, c, h//2, w//2, -1).permute(0, 1, 4, 2, 3).contiguous().view(b, -1, h//2, w//2)
        return x_processed

    def forward(self, x1, x2):
        x1_processed = self.process_tensor(x1)  # (64, 256, 4, 4)
        x2_processed = self.process_tensor(x2)  # (64, 256, 4, 4)
        del x1, x2
        b, c, h, w = x1_processed.shape
        chunk_size = 4  # 每块4个通道

        # 将通道维度拆分为块 [b, num_chunks, chunk_size, h, w]
        x1_chunks = x1_processed.view(b, -1, chunk_size, h, w)  # num_chunks = (FM*4)/4 = FM
        x2_chunks = x2_processed.view(b, -1, chunk_size, h, w)

        # 交替拼接 x1 和 x2 的块 [b, num_chunks*2, chunk_size, h, w]
        combined = torch.stack([x1_chunks, x2_chunks], dim=2).view(b, -1, h, w)  # 通道数 FM*8
        del  x1_processed, x2_processed, x1_chunks, x2_chunks
        output = self.conv1x1(combined)  # (64, 256, 4, 4)
        del combined

        output = self.BN(output)
        output = self.leaky_relu(output)
        output = self.dropout(output)

        return output  # (64, 256, 4, 4)



def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)




class MultiHeadSelfAttention(nn.Module):
    def __init__(self, input_channels, output_channels, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.num_heads = num_heads
        self.d_head = output_channels // num_heads

        # 定义线性投影层
        self.W_q = nn.Linear(input_channels, output_channels)
        self.W_k = nn.Linear(input_channels, output_channels)
        self.W_v = nn.Linear(input_channels, output_channels)
        self.W_o = nn.Linear(output_channels, output_channels)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # 输入形状: (batch_size, input_channels, height, width)
        batch_size, _, height, width = x.size()
        # 将输入转换为二维张量 (batch_size, seq_len, input_channels)
        x = x.view(batch_size, self.input_channels, -1).transpose(1, 2)

        # 线性投影
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # 分割为多个头
        Q = Q.view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_head, dtype=torch.float32))

        # 应用 Softmax 得到注意力权重
        attn_weights = self.softmax(scores)

        # 计算加权和
        output = torch.matmul(attn_weights, V)

        # 拼接多头输出
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.output_channels)

        # 线性变换
        output = self.W_o(output)

        # 将输出转换回四维张量 (batch_size, output_channels, 1, 1)
        output = output.transpose(1, 2).view(batch_size, self.output_channels, height, width)

        return output


def get_MultiHeadSelfAttention(input_channels, output_channels, num_heads):
    return MultiHeadSelfAttention(input_channels=input_channels, output_channels=output_channels, num_heads=num_heads)


class MLP(nn.Module):
    def __init__(self, dim,hidden_size,projection_size):#hidden_size = 2048
        super().__init__()
        self.layer = nn.Sequential(
            #nn.Linear(dim, hidden_size),
            #nn.BatchNorm1d(hidden_size),
            #nn.ReLU(inplace=True),
            #nn.Linear(hidden_size, projection_size)
            nn.Linear(dim, projection_size),
            nn.BatchNorm1d(projection_size),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 输入 x1 的形状为 (batch_size, channels, height, width)，这里假设为 (64, 256, 1, 1)
        batch_size = x.size(0)
        # 先将输入张量从 (batch_size, channels, height, width) 调整为 (batch_size, channels)
        x = x.view(batch_size, -1)
        return self.layer(x)



class pyCNN(nn.Module):
    def __init__(self,NC,Classes,FM=32):
        super(pyCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = NC,out_channels = FM,kernel_size = 3,stride =2,padding = 0),
            nn.BatchNorm2d(FM),
            nn.LeakyReLU(),
            #nn.MaxPool2d(kernel_size=2),
            # nn.Dropout(0.5),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(FM*4, int(FM/2), 4, 2, 1),
            nn.BatchNorm2d(int(FM/2)),
            nn.LeakyReLU(),
            #nn.MaxPool2d(2),
            nn.Dropout(0.5),
        )

        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(FM, int(FM/2), 3, 1, padding=1),
        #     nn.BatchNorm2d(int(FM/2)),
        #     nn.LeakyReLU(),
        #     nn.MaxPool2d(kernel_size=4, stride=4),
        #     nn.Dropout(0.5),
        # )

        self.conv3 = nn.Sequential(
            #get_pyconv(inplans=FM*2, planes=FM*4, pyconv_kernels=[3, 5, 7], stride=1, pyconv_groups=[1, 4, 8]),
            get_MultiHeadSelfAttention(input_channels=FM*2, output_channels=FM*4, num_heads=8),
            nn.BatchNorm2d(FM*4),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.5),
            )

        # self.conv3 = nn.Sequential(
        #     # get_pyconv(inplans=FM*2, planes=FM*4, pyconv_kernels=[3, 5, 7], stride=1, pyconv_groups=[1, 4, 8]),
        #     get_MultiHeadSelfAttention(input_channels=int(FM /2), output_channels=FM * 4, num_heads=8),
        #     nn.BatchNorm2d(FM * 4),
        #     nn.LeakyReLU(),
        #     nn.MaxPool2d(kernel_size=4),
        #     nn.Dropout(0.5),
        # )

        self.conv4 = nn.Sequential(
            nn.Conv2d(1, FM, 3, 2, 0, ),
            nn.BatchNorm2d(FM),
            nn.LeakyReLU(),
            # nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.5),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(FM*4, int(FM/2), 4, 2, 1),
            nn.BatchNorm2d(int(FM/2)),
            nn.LeakyReLU(),
            #nn.MaxPool2d(2),
            nn.Dropout(0.5),
        )

        # self.conv5 = nn.Sequential(
        #     nn.Conv2d(FM, int(FM / 2), 3, 1, 1),
        #     nn.BatchNorm2d(int(FM / 2)),
        #     nn.LeakyReLU(),
        #     nn.MaxPool2d(kernel_size=4, stride=4),
        #     nn.Dropout(0.5),
        # )

        self.conv6 = nn.Sequential(
            #get_pyconv(inplans=FM * 2, planes=FM * 4, pyconv_kernels=[3, 5, 7], stride=1, pyconv_groups=[1, 4, 8]),
            get_MultiHeadSelfAttention(input_channels=FM*2, output_channels=FM * 4, num_heads=8),
            nn.BatchNorm2d(FM * 4),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.5),
        )
        # self.conv6 = nn.Sequential(
        #     # get_pyconv(inplans=FM * 2, planes=FM * 4, pyconv_kernels=[3, 5, 7], stride=1, pyconv_groups=[1, 4, 8]),
        #     get_MultiHeadSelfAttention(input_channels=int(FM /2), output_channels=FM * 4, num_heads=8),
        #     nn.BatchNorm2d(FM * 4),
        #     nn.LeakyReLU(),
        #     nn.MaxPool2d(kernel_size=4),
        #     nn.Dropout(0.5),
        # )

        self.out = nn.Linear(FM*4,Classes)

        self.projection_head1 = MLP(FM*4,FM*64,FM*4)
        self.projection_head2 = MLP(FM*4,FM*64,FM*4)

        self.downsample1 = nn.PixelUnshuffle(2)
        self.downsample2 = nn.PixelUnshuffle(2)
        self.CMFM1 = CMFM(FM)
        self.CMFM2 = CMFM(FM//2)
        self.HGM1 = HyperGroupMix(groups=[[0, 16], [16, 32], [32, 64]],p=0.3)
        self.HGM2 = HyperGroupMix(groups=[[0, 32], [32, 64], [64, 96],[96, 128], [128, 160], [160, 192], [192, 224], [224, 256]],p=0.2)
        self.HGM3 = HyperGroupMix(groups=[[0, 16], [16, 32]],p=0.1)
        # self.MS1 = MixStyle(p=0.6)
        # self.MS2 = MixStyle(p=0.3)
        # self.MS3 = MixStyle(p=0.2)



    def forward(self, x1, x2):

        x1 = self.conv1(x1)
        x2 = self.conv4(x2)
        # print("conv1:",x1.shape)
        # print("conv4:",x2.shape)
        # print("")
        in_1=x1

        x1 = self.HGM1(x1)
        # x1 = self.MS1(x1)
        out_1=x1
        input_CMFPG = x1


        x1 = self.CMFM1(x1,x2)
        x2 = self.downsample1(x2)
        # print("CMFM1:",x1.shape)
        # print("downsample1:",x2.shape)
        in_2 = x1

        x1 = self.HGM2(x1)
        # x1 = self.MS2(x1)
        out_2 = x1

        x1 = self.conv2(x1)
        x2 = self.conv5(x2)
        # print("conv2:",x1.shape)
        # print("conv5:",x2.shape)
        # print("")
        in_3 = x1

        x1 = self.HGM3(x1)
        # x1 = self.MS3(x1)
        out_3 = x1

        x1 = self.CMFM2(x1,x2)
        x2 = self.downsample2(x2)
        # print("CMFM2:",x1.shape)
        # print("downsample2:",x2.shape)
        output_CMFPG = x1

        x1 = self.conv3(x1)
        x2 = self.conv6(x2)
        # print("conv3:",x1.shape)
        # print("conv6:",x2.shape)
        # print("")

        x1 = self.projection_head1(x1)
        x2 = self.projection_head2(x2)

        x = x1 + x2
        out3 = self.out(x)

        return x1, x2, out3, input_CMFPG, output_CMFPG, in_1, in_2, in_3, out_1, out_2, out_3



