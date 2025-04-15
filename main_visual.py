
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
import os
import sys
import numpy as np
import time
from model import *
import torch.optim as optim 
import random
import pdb
import shutil
from LSR import LSR
from torch.cuda.amp import autocast, GradScaler
import pandas as pd
import matplotlib.pyplot as plt

# torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.benchmark = True
parser = argparse.ArgumentParser()

# '''
# 需求：在excel表格中，记录指定轮次的实验最有train和test结果。根据指定的最后一个轮次，画出对应的图像。
# '''
import xlwt

book = xlwt.Workbook(encoding='utf-8')  # 创建Workbook，相当于创建Excel
# 创建sheet，Sheet1为表的名字，cell_overwrite_ok为是否覆盖单元格
sheet1 = book.add_sheet(u'Train_data', cell_overwrite_ok=True)
# 向表中添加数据
sheet1.write(0, 0, 'epoch')
sheet1.write(0, 1, 'Train_Loss')
sheet1.write(0, 2, 'Train_Acc')
sheet1.write(0, 3, 'Val_Loss')
sheet1.write(0, 4, 'Val_Acc')
sheet1.write(0, 5, 'lr')
sheet1.write(0, 6, 'Best val Acc')

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


parser.add_argument('--gpus', type=str, default='0')
parser.add_argument('--lr', type=float, default=0.0003)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--n_class', type=int, default=500)
parser.add_argument('--MAXseqlen', type=int, default=29)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--max_epoch', type=int, default=120)
parser.add_argument('--test', type=str2bool, default=False)

# load opts
parser.add_argument('--weights', type=str, default=None)   # Weight file path, initialization

# save prefix
parser.add_argument('--save_prefix', type=str, default='F:/code/pythoncode/lip/checkpoints/3dresnet18-DCTCN-skipconnection-all-20241009/')   # weight file path

# dataset
parser.add_argument('--dataset', type=str, default='lrw')
parser.add_argument('--border', type=str2bool, default=True)
parser.add_argument('--mixup', type=str2bool, default=True)
parser.add_argument('--label_smooth', type=str2bool, default=True)
parser.add_argument('--se', type=str2bool, default=True)

'''
python main_visual.py \
    --gpus='0,1,2,3'  \
    --lr=3e-4 \
    --batch_size=400 \
    --num_workers=8 \
    --max_epoch=120 \
    --test=False \
    --save_prefix='checkpoints/lrw-baseline/' \
    --n_class=500 \
    --dataset='lrw' \
    --border=False \
    --mixup=False \
    --label_smooth=False \
    --se=False  
'''

# 是否TM :yes

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus


torch.manual_seed(1)   # 种子参数
np.random.seed(1)
random.seed(1)

if (args.dataset == 'lrw'):
    from utils import LRWDataset as Dataset
elif (args.dataset == 'lrw1000'):
    from utils import LRW1000_Dataset as Dataset
else:
    raise Exception('lrw or lrw1000')

video_model = VideoModel(args).cuda().to(torch.float32)


def parallel_model(model):
    model = nn.DataParallel(model)
    return model


def load_missing(model, pretrained_dict):
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                       k in model_dict.keys() and v.size() == model_dict[k].size()}
    missed_params = [k for k, v in model_dict.items() if  k not in pretrained_dict.keys()]

    print('loaded params/tot params:{}/{}'.format(len(pretrained_dict), len(model_dict)))
    print('miss matched params:', missed_params)
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


lr = args.batch_size / 32.0 / torch.cuda.device_count() * args.lr
optim_video = optim.AdamW(video_model.parameters(), lr=lr, weight_decay=1e-2)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optim_video, T_max=args.max_epoch, eta_min=5e-6)

if (args.weights is not None):
    print('load weights')
    weight = torch.load(args.weights, map_location=torch.device('cpu'))
    load_missing(video_model, weight.get('video_model'))

# test flops时注释掉下面一行
video_model = parallel_model(video_model)


def dataset2dataloader(dataset, batch_size, num_workers, shuffle=True):
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        shuffle=shuffle,
                        drop_last=False,
                        pin_memory=True)
    return loader


def add_msg(msg, k, v):
    if (msg != ''):
        msg = msg + ','
    msg = msg + k.format(v)
    return msg

# 使用NumPy创建全局大小为500的数组并初始化为0
# global_array = np.zeros(500)
def test():
    with torch.no_grad():
        dataset = Dataset('test', args)
        print('Start Testing, Data Length:', len(dataset))
        loader = dataset2dataloader(dataset, args.batch_size, args.num_workers, shuffle=False)
        print('start testing')
        v_acc = []
        entropy = []
        acc_mean = []
        total = 0
        cons_acc = 0.0
        cons_total = 0.0
        attns = []

        for (i_iter, input) in enumerate(loader):

            video_model.eval()

            tic = time.time()
            video = input.get('video').cuda(non_blocking=True)

            label = input.get('label').cuda(non_blocking=True)
            # print(label[0])

            # exit(0)
            total = total + video.size(0)
            names = input.get('name')
            border = input.get('duration').cuda(non_blocking=True).float()
            # mask = input.get('mask').cuda(non_blocking=True)
            # print(border[0])
            with autocast():
                if (args.border):
                    y_v = video_model(video, border=border)  # , mask=mask
                else:
                    y_v = video_model(video)  # , mask=mask

            v_acc.extend((y_v.argmax(-1) == label).cpu().numpy().tolist())
            toc = time.time()
            if (i_iter % 10 == 0):
                msg = ''
                msg = add_msg(msg, 'v_acc={:.5f}', np.array(v_acc).reshape(-1).mean())
                msg = add_msg(msg, 'eta={:.5f}', (toc - tic) * (len(loader) - i_iter) / 3600.0)

                print(msg)

        acc = float(np.array(v_acc).reshape(-1).mean())
        msg = 'v_acc_{:.5f}_'.format(acc)
        # print(global_array)

        return acc, msg

def showLR(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += ['{:.6f}'.format(param_group['lr'])]
    return ','.join(lr)
def train():

        dataset = Dataset('train', args)
        print('Start Training, Data Length:', len(dataset))

        loader = dataset2dataloader(dataset, args.batch_size, args.num_workers)

        max_epoch = args.max_epoch

        ce = nn.CrossEntropyLoss()

        tot_iter = 0
        best_acc = 0.0
        adjust_lr_count = 0
        alpha = 0.2
        beta_distribution = torch.distributions.beta.Beta(alpha, alpha)
        scaler = GradScaler()
        for epoch in range(max_epoch):
            sheet1.write(epoch + 1, 0, epoch + 1)
            total = 0.0
            v_acc = 0.0
            total = 0.0

            lsr = LSR()

            for (i_iter, input) in enumerate(loader):
                tic = time.time()

                video_model.train()
                video = input.get('video').cuda(non_blocking=True).clone()

                label = input.get('label').cuda(non_blocking=True).long().clone()
                border = input.get('duration').cuda(non_blocking=True).float().clone()

                loss = {}

                if (args.label_smooth):
                    loss_fn = lsr
                else:
                    loss_fn = nn.CrossEntropyLoss()

                with autocast():
                    if (args.mixup):
                        lambda_ = np.random.beta(alpha, alpha)
                        index = torch.randperm(video.size(0)).cuda(non_blocking=True)

                        mix_video = lambda_ * video + (1 - lambda_) * video[index, :]
                        mix_border = lambda_ * border + (1 - lambda_) * border[index, :]


                        # print(video.get(0).size())
                        label_a, label_b = label, label[index]

                        if (args.border):
                            y_v = video_model(mix_video, border=mix_border)  # , mask=mask
                        else:
                            y_v = video_model(mix_video)  # , mask=mask
                        alpha = y_v + 1

                        loss_bp = lambda_ * loss_fn(y_v, label_a) + (1 - lambda_) * loss_fn(y_v, label_b)

                    else:
                        if (args.border):
                            y_v = video_model(video, border=border)  # , mask=mask
                        else:
                            y_v = video_model(video)  # , mask=mask

                        loss_bp = loss_fn(y_v, label)

                loss['CE V'] = loss_bp

                optim_video.zero_grad()
                scaler.scale(loss_bp).backward()
                scaler.step(optim_video)
                scaler.update()

                toc = time.time()

                msg = 'epoch={},train_iter={},eta={:.5f}'.format(epoch, tot_iter,
                                                                 (toc - tic) * (len(loader) - i_iter) / 3600.0)
                for k, v in loss.items():
                    msg += ',{}={:.5f}'.format(k, v)
                msg = msg + str(',lr=' + str(showLR(optim_video)))
                msg = msg + str(',best_acc={:2f}'.format(best_acc))
                print(msg)

                if (i_iter == len(loader) - 1 or (epoch == 0 and i_iter == 0)):

                    acc, msg = test()
                    sheet1.write(epoch + 1, 4, str(acc))
                    sheet1.write(epoch + 1, 5, str(showLR(optim_video)))
                    if (acc > best_acc):
                        savename = '{}_iter_{}_epoch_{}_{}.pt'.format(args.save_prefix, tot_iter, epoch, msg)

                        temp = os.path.split(savename)[0]
                        if (not os.path.exists(temp)):
                            os.makedirs(temp)
                        torch.save(
                            {
                                'video_model': video_model.module.state_dict(),
                            }, savename)

                    if (tot_iter != 0):
                        best_acc = max(acc, best_acc)
                        sheet1.write(epoch + 1, 6, str(best_acc))

                tot_iter += 1

            scheduler.step()
        book.save('.\Train_data.xlsx')
        # 可视化展示结果
        # 读取 Excel 文件
        file_path = '.\Train_data.xlsx'  # 更改为你的 Excel 文件路径
        data = pd.read_excel(file_path)




if(__name__ == '__main__'):
    
    if(args.test):
        acc, msg = test()
        print(f'acc={acc}')
        # exit()
    train()

    # mask = torch.ones(1, 29, 1, 88, 88).to('cuda:0')
    # border_mask = torch.ones(1, 29).to('cuda:0')
    # print('==> Building model..')
    # from thop import profile
    # flops, params = profile(video_model, (mask, border_mask))
    # print('flops: ', flops, 'params:', params)
    # print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))






























# import argparse
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# import math
# import os
# import sys
# import numpy as np
# import time
# from model import *
# import torch.optim as optim 
# import random
# import pdb
# import shutil
# from LSR import LSR
# from torch.cuda.amp import autocast, GradScaler


# torch.backends.cudnn.benchmark = True
# parser = argparse.ArgumentParser()
# CUDA_LAUNCH_BLOCKING=1

# def str2bool(v):
#     if v.lower() in ('yes', 'true', 't', 'y', '1'):
#         return True
#     elif v.lower() in ('no', 'false', 'f', 'n', '0'):
#         return False
#     else:
#         raise argparse.ArgumentTypeError('Unsupported value encountered.')

# parser.add_argument('--gpus', type=str,default='0')
# parser.add_argument('--lr', type=float,default='0.0003')
# parser.add_argument('--batch_size', type=int,default='64')
# parser.add_argument('--n_class', type=int, default='500')
# parser.add_argument('--num_workers', type=int, default='2')
# parser.add_argument('--max_epoch', type=int, default='120')
# parser.add_argument('--test', type=str2bool, default='False')

# # load opts
# parser.add_argument('--weights', type=str,  default='/root/autodl-tmp/lrw-baseline1_iter_443003_epoch_115_v_acc_0.40360_.pt')

# # save prefix
# parser.add_argument('--save_prefix', type=str, default='/root/learn-an-effective-lip-reading-model-without-pains-master/learn-an-effective-lip-reading-model-without-pains-master/checkpoints/lrw-baseline-encoder/')

# # dataset
# parser.add_argument('--dataset', type=str, default='lrw')
# parser.add_argument('--border', type=str2bool,default='False')
# parser.add_argument('--mixup', type=str2bool, default='False')
# parser.add_argument('--label_smooth', type=str2bool, default='False')
# parser.add_argument('--se', type=str2bool, default='False')


# args = parser.parse_args()
# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

# if(args.dataset == 'lrw'):
#     from utils import LRWDataset as Dataset
# elif(args.dataset == 'lrw1000'):    
#     from utils import LRW1000_Dataset as Dataset
# else:
#     raise Exception('lrw or lrw1000')    


# video_model = VideoModel(args).cuda()

# def parallel_model(model):
#     model = nn.DataParallel(model)
#     return model        


# def load_missing(model, pretrained_dict):
#     model_dict = model.state_dict()
#     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys() and v.size() == model_dict[k].size()}                
#     missed_params = [k for k, v in model_dict.items() if not k in pretrained_dict.keys()]
    
#     print('loaded params/tot params:{}/{}'.format(len(pretrained_dict),len(model_dict)))
#     print('miss matched params:',missed_params)
#     model_dict.update(pretrained_dict)
#     model.load_state_dict(model_dict)
#     return model
    

# lr = args.batch_size / 32.0 / torch.cuda.device_count() * args.lr
# optim_video = optim.Adam(video_model.parameters(), lr = lr, weight_decay=1e-4)     
# scheduler = optim.lr_scheduler.CosineAnnealingLR(optim_video, T_max = args.max_epoch, eta_min=5e-6)


# if(args.weights is not None):
#     print('load weights')
#     print(args.weights)
#     weight = torch.load(args.weights, map_location=torch.device('cpu'))  
#     # print('----------------')  
#     load_missing(video_model, weight.get('video_model'))
    
          
# video_model = parallel_model(video_model)

# def dataset2dataloader(dataset, batch_size, num_workers, shuffle=True):
#     loader =  DataLoader(dataset,
#             batch_size = batch_size, 
#             num_workers = num_workers,   
#             shuffle = shuffle,         
#             drop_l_ast = False,
#             pin_memory=True)
#     return loader

# def add_msg(msg, k, v):
#     if(msg != ''):
#         msg = msg + ','
#     msg = msg + k.format(v)
#     return msg    

# def test():
    
#     with torch.no_grad():
#         dataset = Dataset('val', args)
#         print('Start Testing, Data Length:',len(dataset))
#         loader = dataset2dataloader(dataset, args.batch_size, args.num_workers, shuffle=False)        
        
#         print('start testing')
#         v_acc = []
#         entropy = []
#         acc_mean = []
#         total = 0
#         cons_acc = 0.0
#         cons_total = 0.0
#         attns = []

#         for (i_iter, input) in enumerate(loader):
            
#             video_model.eval()
            
#             tic = time.time()
#             video = input.get('video').cuda(non_blocking=True)
#             label = input.get('label').cuda(non_blocking=True)
#             total = total + video.size(0)
#             names = input.get('name')
#             border = input.get('duration').cuda(non_blocking=True).float()
            
#             with autocast():
#                 if(args.border):
#                     y_v = video_model(video, border)                                           
#                 else:
#                     y_v = video_model(video)                                           
                                

#             v_acc.extend((y_v.argmax(-1) == label).cpu().numpy().tolist())
#             toc = time.time()
#             if(i_iter % 10 == 0):  
#                 msg = ''              
#                 msg = add_msg(msg, 'v_acc={:.5f}', np.array(v_acc).reshape(-1).mean())                
#                 msg = add_msg(msg, 'eta={:.5f}', (toc-tic)*(len(loader)-i_iter)/3600.0)
                                
#                 print(msg)            

#         acc = float(np.array(v_acc).reshape(-1).mean())
#         msg = 'v_acc_{:.5f}_'.format(acc)
        
#         return acc, msg                                 

# def showLR(optimizer):
#     lr = []
#     for param_group in optimizer.param_groups:
#         lr += ['{:.6f}'.format(param_group['lr'])]
#     return ','.join(lr)

# def train():            
    
    
    
#     dataset = Dataset('train', args)
#     print('Start Training, Data Length:',len(dataset))
    
#     loader = dataset2dataloader(dataset, args.batch_size, args.num_workers)
        
#     max_epoch = args.max_epoch    

    
#     ce = nn.CrossEntropyLoss()

#     tot_iter = 0
#     best_acc = 0.0
#     adjust_lr_count = 0
#     alpha = 0.2
#     beta_distribution = torch.distributions.beta.Beta(alpha, alpha)
#     scaler = GradScaler()             
#     for epoch in range(max_epoch):
#         total = 0.0
#         v_acc = 0.0
#         total = 0.0               
        
#         lsr = LSR()
        
#         print('total------------------')
#         print(len(loader))
#         # print('-------------------------------')
#         for (i_iter, input) in enumerate(loader):
#             tic = time.time()           
            
#             video_model.train()
#             video = input.get('video').cuda(non_blocking=True)
#             label = input.get('label').cuda(non_blocking=True).long()     
#             border = input.get('duration').cuda(non_blocking=True).float()
            
#             loss = {}
            
#             if(args.label_smooth):
#                 loss_fn = lsr
#             else:
#                 loss_fn = nn.CrossEntropyLoss()
            
#             with autocast():
#                 if(args.mixup):
#                     lambda_ = np.random.beta(alpha, alpha)
#                     index = torch.randperm(video.size(0)).cuda(non_blocking=True)
                    
#                     mix_video = lambda_ * video + (1 - lambda_) * video[index, :]
#                     mix_border = lambda_ * border + (1 - lambda_) * border[index, :]
                        
#                     label_a, label_b = label, label[index]            

#                     if(args.border):
#                         y_v = video_model(mix_video, mix_border)       
#                     else:                
#                         y_v = video_model(mix_video)       

#                     loss_bp = lambda_ * loss_fn(y_v, label_a) + (1 - lambda_) * loss_fn(y_v, label_b)
                    
#                 else:
#                     if(args.border):
#                         y_v = video_model(video, border)       
#                     else:                
#                         y_v = video_model(video)    
                        
#                     loss_bp = loss_fn(y_v, label)
                                    
            
#             loss['CE V'] = loss_bp
                
#             optim_video.zero_grad()   
#             scaler.scale(loss_bp).backward()  
#             scaler.step(optim_video)
#             scaler.update()
            
#             toc = time.time()
            
#             msg = 'epoch={},train_iter={},eta={:.5f}'.format(epoch, tot_iter, (toc-tic)*(len(loader)-i_iter)/3600.0)
#             for k, v in loss.items():                                                
#                 msg += ',{}={:.5f}'.format(k, v)
#             msg = msg + str(',lr=' + str(showLR(optim_video)))                    
#             msg = msg + str(',best_acc={:2f}'.format(best_acc))
#             print(msg)                                
            
#             if(i_iter == len(loader) - 1 or (epoch == 0 and i_iter == 0)):

#                 acc, msg = test()

#                 if(acc > best_acc):
#                     savename = '{}_iter_{}_epoch_{}_{}.pt'.format(args.save_prefix, tot_iter, epoch, msg)
                    
#                     temp = os.path.split(savename)[0]
#                     if(not os.path.exists(temp)):
#                         os.makedirs(temp)                    
#                     torch.save(
#                         {
#                             'video_model': video_model.module.state_dict(),
#                         }, savename)         
                     

#                 if(tot_iter != 0):
#                     best_acc = max(acc, best_acc)    
                    
#             tot_iter += 1        
            
#         scheduler.step()            
        
# if(__name__ == '__main__'):
#     print(model)
#     if(args.test):
#         acc, msg = test()
#         print(f'acc={acc}')
#         exit()
    
#     train()

