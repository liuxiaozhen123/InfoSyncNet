# coding: utf-8
import math
import numpy as np
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from model.VisionTransformer import VisionTransformer
from utils.OT import SinkhornDistance


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, se=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.se = se
        
        if(self.se):
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.conv3 = conv1x1(planes, planes//16)
            self.conv4 = conv1x1(planes//16, planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
            
        if(self.se):
            w = self.gap(out)
            w = self.conv3(w)
            w = self.relu(w)
            w = self.conv4(w).sigmoid()
            
            out = out * w
        
        out = out + residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, se=False):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.se = se
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        self.bn = nn.BatchNorm1d(512)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, se=self.se))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, se=self.se))

        return nn.Sequential(*layers)

    # def forward(self, x):
    #     x = self.layer1(x)
    #     x = self.layer2(x)
    #     x = self.layer3(x)
    #     x = self.layer4(x)
    #     x = self.avgpool(x)
    #     x = x.view(x.size(0), -1)
    #     x = self.bn(x)
    #     return x       
    def forward(self, x):
        # print("-------------resnet18 begin--------------")
        # print(x.size())  # 928 64 22 22
        x = self.layer1(x)
        # print(x.size())  # 928 64 22 22
        x = self.layer2(x)
        # print(x.size())  # 928 128 11 11
        x = self.layer3(x)
        # print(x.size())  # 928 256 6 6
        x = self.layer4(x)
        # print(x.size())  # 928 512 3 3
        # print(type(x))
        # print(x.shape)
        # t_a = target
        # t_b = target
        # if mode == 'train':
        #     x, t_a, t_b = mixup_aligned(x, target, lam)
        # x = x - 0.00001
        # print(x.shape)
        # print(type(x))
        x = self.avgpool(x)
        # print(x.size())  # 928 512 1 1
        x = x.view(x.size(0), -1)
        # print(x.size())  # 928 512
        x = self.bn(x)
        # print(x.size())   # 928 512
        # print("-----------------------resnet18 end-------------------")
        return x
 

'''接入vit后进行修改'''


def mixup_aligned(out, y, lam):
    # out shape = batch_size x 512 x 4 x 4 (cifar10/100)
    # print("-------------------------out------------------------")
    # indices = np.random.permutation(out.size(0))  # 生成一个随机索引 序列，用于从批次中选择样本进行混合
    indices_init = np.random.permutation(int(out.size(0)/29))

    indices = np.repeat(indices_init*29, 29)
    increments = np.tile(np.arange(29), int(out.size(0)/29))
    indices = indices + increments
    feat1 = out.view(out.shape[0], out.shape[1], -1) # batch_size x 512 x 16
    # print("feat1")
    # print(feat1.size())
    feat2 = out[indices].view(out.shape[0], out.shape[1], -1) # batch_size x 512 x 16 使用随机索引重塑输出张量，用于混合
    # print("feat2")
    # print(feat2.size())
    sinkhorn = SinkhornDistance(eps=0.1, max_iter=100, reduction=None)  # 使用S算法计算两组特征之间的最优传输矩阵 eps:正则化系数 max_iter:最大迭代次数，防止算法在难以收敛的情况下无限运行 reduction:指定输出降维的方式
    P = sinkhorn(feat1.permute(0,2,1), feat2.permute(0,2,1)).detach()  # optimal plan batch x 16 x 16
    
    P = P*(out.size(2)*out.size(3)) # assignment matrix 

    align_mix = random.randint(0,1) # uniformly choose at random, which alignmix to perform
   
    if (align_mix == 0):    # 两种混合方式
        # \tilde{A} = A'R^{T}
        f1 = torch.matmul(feat2, P.permute(0,2,1).cuda()).view(out.shape) 
        final = feat1.view(out.shape)*lam + f1*(1-lam)

    elif (align_mix == 1):
        # \tilde{A}' = AR
        f2 = torch.matmul(feat1, P.cuda()).view(out.shape).cuda()
        final = f2*lam + feat2.view(out.shape)*(1-lam)

    y_a, y_b = y,y[indices_init]

    return final, y_a, y_b

def mixup_process(out, y, lam):
        indices = np.random.permutation(out.size(0))
        out = out * lam + out[indices] * (1 - lam)
        y_a, y_b = y, y[indices]
        return out, y_a, y_b

class VideoCNN(nn.Module):
    def __init__(self, se=False):
        super(VideoCNN, self).__init__()
        
        # frontend3D
        self.frontend3D = nn.Sequential(
                nn.Conv3d(1, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
                nn.BatchNorm3d(64),
                nn.ReLU(True),
                nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
                )
        # resnet

        self.resnet18 = ResNet(BasicBlock, [2, 2, 2, 2], se=se)  # 3 4 6 3
        # self.vit_block = VisionTransformer()
        self.dropout = nn.Dropout(p=0.5)
        # self.vit = VisionTransformer()

        # backend_gru
        self._initialize_weights()
        # for m in self.modules():
        #     print(m)

    
    def visual_frontend_forward(self, x):
        # layer_mix = random.randint(0, 1)
        # global ans_label_a, ans_label_b
        # layer_mix = 1
        # print(layer_mix)
        # if mode == 'train':
        #     if layer_mix == 0:
        #         x, ans_label_a, ans_label_b = mixup_process(x, target, lam)
        #         # print(x.size())  # 32 29 1 88 88
        #         x = x.transpose(1, 2)
        #         # print(x.size())  # 32 1 29 88 88
        #         x = self.frontend3D(x)
        #         # print(x.size())  # 32 64 29 22 22
        #         x = x.transpose(1, 2)
        #         # print(x.size())  # 32 29 64 22 22
        #         x = x.contiguous()
        #         x = x.view(-1, 64, x.size(3), x.size(4))
        #         # print(x.size())  # 928 64 22 22
        #         mode = None
        #         x, a, b = self.resnet18(x, target, lam, mode)
        #     elif layer_mix == 1:
        #         # print(x.size())  # 32 29 1 88 88
        #         x = x.transpose(1, 2)
        #         # print(x.size())  # 32 1 29 88 88
        #         x = self.frontend3D(x)
        #         # print(x.size())  # 32 64 29 22 22
        #         x = x.transpose(1, 2)
        #         # print(x.size())  # 32 29 64 22 22
        #         x = x.contiguous()
        #         x = x.view(-1, 64, x.size(3), x.size(4))
        #         # print(x.size())  # 928 64 22 22
        #         x, ans_label_a, ans_label_b= self.resnet18(x, target, lam, mode)
        # # print(x.size())  # 928 512
        # # print("-----------------------end 3dcnn + resnet18")
        # else:
        #     x = x.transpose(1, 2)
        #     # print(x.size())  # 32 1 29 88 88
        #     x = self.frontend3D(x)
        #     # print(x.size())  # 32 64 29 22 22
        #     x = x.transpose(1, 2)
        #     # print(x.size())  # 32 29 64 22 22
        #     x = x.contiguous()
        #     x = x.view(-1, 64, x.size(3), x.size(4))
        #     x, ans_label_a, ans_label_b = self.resnet18(x, target, lam, mode)
        # print("----------------begin 3DCNN+ resnet18-----------")
        # print(x.size())  # 32 29 1 88 88
        x = x.transpose(1, 2)
        # print(x.size())  # 32 1 29 88 88
        x = self.frontend3D(x)
        # print(x.size())  # 32 64 29 22 22
        x = x.transpose(1, 2)
        # print(x.size())  # 32 29 64 22 22
        x = x.contiguous()
        x = x.view(-1, 64, x.size(3), x.size(4))
        # print(x.size())  # 928 64 22 22
        # x,ans_label_a,ans_label_b = self.resnet18(x)
        x = self.resnet18(x)
        # print(x.size())  # 928 512
        # print("-----------------------end 3dcnn + resnet18")
        return x
    # def visual_frontend_forward(self, x):
    #     # print("----------------lam-----------------------")
    #     # print(lam)
    #     x = x.transpose(1, 2)
    #     x = self.frontend3D(x)
    #     x = x.transpose(1, 2)
    #     # print(x.size())
    #     x = x.contiguous()
    #     x = x.view(-1, 64, x.size(3), x.size(4))
    #     # print(x.size())
    #     # xr = self.resnet18(x)
    #     # xv = self.vit_block(x)  # [- in_channels image_size image_size]
    #     # x = torch.cat((xr, xv), dim=0)
    #     # '''尝试添加alignmixup'''
    #     # if(mode == 'train'):
    #     #     x,t_a,t_b = mixup_aligned(x)    
    #     # print(x.size())
    #     x = self.resnet18(x)
    #     # print(x.size())
    #     return x        

    def forward(self, x):
        b, t = x.size()[:2]

        x = self.visual_frontend_forward(x)

        # x = self.dropout(x)
        feat = x.view(b, -1, 512)

        # x = x.view(b*t, 2, 16, 16)
        # x = self.vit(x)
        # x = x.view(b, t, 128)
        x = x.view(b, t, 512)
        # print(x.size())
        return x
    # def forward(self, x, target, lam, mode):
    #     # print(x)
    #     b, t = x.size()[:2]

    #     x,ans_label_a,ans_label_b = self.visual_frontend_forward(x, target, lam, mode)
        
    #     #x = self.dropout(x)
    #     feat = x.view(b, -1, 512)

    #     x = x.view(b, -1, 512)  # B, T, C512       
    #     return x,ans_label_a,ans_label_b

    def _initialize_weights(self):
        for m in self.modules():
            # print(m)
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()











# coding: utf-8
# import math
# import numpy as np


# import torch
# import torch.nn as nn
# from torch.autograd import Variable
# import torch.nn.functional as F

# def conv3x3(in_planes, out_planes, stride=1):
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                      padding=1, bias=False)


# def conv1x1(in_planes, out_planes, stride=1):
#     return nn.Conv2d(in_planes, out_planes, kernel_size=1)


# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, inplanes, planes, stride=1, downsample=None, se=False):
#         super(BasicBlock, self).__init__()
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.downsample = downsample
#         self.stride = stride
#         self.se = se
        
#         if(self.se):
#             self.gap = nn.AdaptiveAvgPool2d(1)
#             self.conv3 = conv1x1(planes, planes//16)
#             self.conv4 = conv1x1(planes//16, planes)

#     def forward(self, x):
#         residual = x
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         out = self.bn2(out)
        
#         if self.downsample is not None:
#             residual = self.downsample(x)
            
#         if(self.se):
#             w = self.gap(out)
#             w = self.conv3(w)
#             w = self.relu(w)
#             w = self.conv4(w).sigmoid()
            
#             out = out * w
        
#         out = out + residual
#         out = self.relu(out)

#         return out


# class ResNet(nn.Module):

#     def __init__(self, block, layers, se=False):
#         self.inplanes = 64
#         super(ResNet, self).__init__()
#         self.se = se
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        
#         self.avgpool = nn.AdaptiveAvgPool2d(1)
        
#         self.bn = nn.BatchNorm1d(512)
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.BatchNorm1d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()

#     def _make_layer(self, block, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.inplanes, planes * block.expansion,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(planes * block.expansion),
#             )

#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample, se=self.se))
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes, se=self.se))

#         return nn.Sequential(*layers)

#     def forward(self, x):
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         x = self.bn(x)
#         return x        


# global number
# number = 0
# class VideoCNN(nn.Module):
#     def __init__(self, se=False):
#         super(VideoCNN, self).__init__()
        
#         # frontend3D
#         self.frontend3D = nn.Sequential(
#                 nn.Conv3d(1, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
#                 nn.BatchNorm3d(64),
#                 nn.ReLU(True),
#                 nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
#                 )
#         # resnet
#         self.resnet18 = ResNet(BasicBlock, [2, 2, 2, 2], se=se)
#         self.dropout = nn.Dropout(p=0.5)

#         # backend_gru
#         # initialize
#         # print("-------------------------------")
#         # print(number+1)
#         self._initialize_weights()
#         # print("********************************")
    
#     def visual_frontend_forward(self, x):
#         x = x.transpose(1, 2)
#         x = self.frontend3D(x)
#         x = x.transpose(1, 2)
#         x = x.contiguous()
#         x = x.view(-1, 64, x.size(3), x.size(4))
#         x = self.resnet18(x)
#         return x        
    
#     def forward(self, x):
#         b, t = x.size()[:2]

#         x = self.visual_frontend_forward(x)
        
#         #x = self.dropout(x)
#         feat = x.view(b, -1, 512)

#         x = x.view(b, -1, 512)       
#         return x

#     def _initialize_weights(self):
#         for m in self.modules():
#             # print(m)
#             if isinstance(m, nn.Conv3d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#                 if m.bias is not None:
#                     m.bias.data.zero_()

#             elif isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#                 if m.bias is not None:
#                     m.bias.data.zero_()

#             elif isinstance(m, nn.Conv1d):
#                 n = m.kernel_size[0] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#                 if m.bias is not None:
#                     m.bias.data.zero_()

#             elif isinstance(m, nn.BatchNorm3d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()

#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()

#             elif isinstance(m, nn.BatchNorm1d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()


# 注释版本
# def conv3x3(in_planes, out_planes, stride=1):
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                      padding=1, bias=False)


# def conv1x1(in_planes, out_planes, stride=1):
#     return nn.Conv2d(in_planes, out_planes, kernel_size=1)


# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, inplanes, planes, stride=1, downsample=None, se=False):
#         super(BasicBlock, self).__init__()
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.downsample = downsample
#         self.stride = stride
#         self.se = se
        
#         if(self.se):
#             self.gap = nn.AdaptiveAvgPool2d(1)
#             self.conv3 = conv1x1(planes, planes//16)
#             self.conv4 = conv1x1(planes//16, planes)

#     def forward(self, x):
#         residual = x
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         out = self.bn2(out)
        
#         if self.downsample is not None:
#             residual = self.downsample(x)
            
#         if(self.se):
#             w = self.gap(out)
#             w = self.conv3(w)
#             w = self.relu(w)
#             w = self.conv4(w).sigmoid()
            
#             out = out * w
        
#         out = out + residual
#         out = self.relu(out)

#         return out


# class ResNet(nn.Module):

#     def __init__(self, block, layers, se=False):
#         self.inplanes = 64
#         super(ResNet, self).__init__()
#         self.se = se
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        
#         self.avgpool = nn.AdaptiveAvgPool2d(1)
        
#         self.bn = nn.BatchNorm1d(512)
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.BatchNorm1d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()

#     def _make_layer(self, block, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:  # 判断是否进行下采样操作，保证输入和输出维度相同
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.inplanes, planes * block.expansion,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(planes * block.expansion),
#             )

#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample, se=self.se))
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes, se=self.se))

#         return nn.Sequential(*layers)

#     def forward(self, x):
#         print("resnet18 输入：")
#         print(x.size())
#         x = self.layer1(x)
#         print("resnet18 1层后：")
#         print(x.size())
#         x = self.layer2(x)
#         print("resnet18 2层后：")
#         print(x.size())
#         x = self.layer3(x)
#         print("resnet18 3层后：")
#         print(x.size())
#         x = self.layer4(x)
#         print("resnet18 4层后：")
#         print(x.size())
#         x = self.avgpool(x)
#         print("resnet18 平均池化后：")
#         print(x.size())
#         x = x.view(x.size(0), -1)
#         print("resnet18 ：将维度变成2维后")
#         print(x.size())
#         x = self.bn(x)
#         print("resnet18 归一化后：")
#         print(x.size())
#         return x        


# class VideoCNN(nn.Module):
#     def __init__(self, se=False):
#         super(VideoCNN, self).__init__()
        
#         # frontend3D
#         self.frontend3D = nn.Sequential(
#                 nn.Conv3d(1, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
#                 nn.BatchNorm3d(64),
#                 nn.ReLU(True),
#                 nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
#                 )
#         # resnet
#         self.resnet18 = ResNet(BasicBlock, [2, 2, 2, 2], se=se)
#         self.dropout = nn.Dropout(p=0.5)

#         # backend_gru
#         # initialize
#         self._initialize_weights()
    
#     def visual_frontend_forward(self, x):
#         x = x.transpose(1, 2) # 将输入的x的维度进行转置，比如（batch_size,C,T,H,W）->(batch_size,T,C,H,W)
#         x = self.frontend3D(x)
#         x = x.transpose(1, 2)
#         x = x.contiguous()  # 将张量x进行内存的整理，以便后续视图操作
#         x = x.view(-1, 64, x.size(3), x.size(4))  # 对张量进行形状变换
#         x = self.resnet18(x)
#         #print("resnet18"+x.size())
#         return x        
    
#     def forward(self, x):
#         print("---------输入3DCNN前")
#         print(x.size())
#         b, t = x.size()[:2]
#         print("b=")
#         print(b)
#         print("t=")
#         print(t)
#         print("---------输入3DCNN前，改变维度后")
#         print(x.size())
#         x = self.visual_frontend_forward(x)
        
#         #x = self.dropout(x)
#         print("---------输入3DCNN后")
#         print(x.size())
#         feat = x.view(b, -1, 512)
#         print("---------输入3DCNN后，改变维度后：")
#         print(x.size())
#         x = x.view(b, -1, 512)       
#         return x

#     def _initialize_weights(self):
#         for m in self.modules():
#             print(m)
#             if isinstance(m, nn.Conv3d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#                 if m.bias is not None:
#                     m.bias.data.zero_()

#             elif isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#                 if m.bias is not None:
#                     m.bias.data.zero_()

#             elif isinstance(m, nn.Conv1d):
#                 n = m.kernel_size[0] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#                 if m.bias is not None:
#                     m.bias.data.zero_()

#             elif isinstance(m, nn.BatchNorm3d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()

#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()

#             elif isinstance(m, nn.BatchNorm1d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
