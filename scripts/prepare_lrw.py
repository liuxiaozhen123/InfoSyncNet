# encoding: utf-8
import cv2
from turbojpeg import TurboJPEG, TJPF_GRAY, TJSAMP_GRAY, TJFLAG_PROGRESSIVE
import torch

import numpy as np
import glob
import time
import cv2
import os
from torch.utils.data import Dataset, DataLoader
import torch
import matplotlib.pyplot as plt

jpeg = TurboJPEG()
def extract_opencv(filename, label_dir):
    video = []
    cap = cv2.VideoCapture(filename)
    frame_count = 0
    # print(filename)
    # print(label_dir)
    save_dir = 'F:\dataset\lrw\lrw_roi_80_116_175_211_npy_gray_pkl_jpeg' + str(label_dir)  # Replace with the location of the preprocessed dataset (set it yourself)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    while(cap.isOpened()):
        ret, frame = cap.read() # BGR
        if ret:
            frame = frame[115:211, 79:175]

            # # 可视化每一帧
            # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转换为RGB格式，适应matplotlib显示
            # plt.imshow(frame_rgb)
            # plt.title(f'Frame {frame_count}')
            # plt.show()
            # 保存每一帧到相应的标签文件夹
            save_path = os.path.join(save_dir, f'frame_{frame_count}.jpg')
            cv2.imwrite(save_path, frame)  # 保存帧图像
            print(f'Saved: {save_path}')

            frame_count += 1
            frame = jpeg.encode(frame)
            video.append(frame)
        else:
            break
    cap.release()
    return video        


target_dir = 'F:\dataset\lrw\lrw_roi_80_116_175_211_npy_gray_pkl_jpeg_easy_to_err'  # Replace with the location of the preprocessed dataset (set it yourself)

if(not os.path.exists(target_dir)):
    os.makedirs(target_dir)    

class LRWDataset(Dataset):
    def __init__(self):

        with open('F:\learn-an-effective-lip-reading-model-without-pains-master\learn-an-effective-lip-reading-model-without-pains-master\label_sorted.txt') as myfile:  #  Replace with the file path of label_sorted.txt
            self.labels = myfile.read().splitlines()            
        
        self.list = []

        for (i, label) in enumerate(self.labels):
            files = glob.glob(os.path.join('F:\dataset\lrw_easy_to_err', label, '*', '*.mp4'))  # Replace with the original dataset path
            for file in files:
                savefile = file.replace('F:\dataset\lrw_easy_to_err', target_dir).replace('.mp4', '.pkl')  # Replace with the original dataset path
                savepath = os.path.split(savefile)[0]
                if(not os.path.exists(savepath)):
                    os.makedirs(savepath)
                
            files = sorted(files)
            

            self.list += [(file, i) for file in files]                                                                                
            
        
    def __getitem__(self, idx):
            
        inputs = extract_opencv(self.list[idx][0], self.list[idx][1])  # 从self.list中 提取当前索引 idx 对应的视频文件路径。 [0]位置即存储文件路径
        result = {}        
         
        name = self.list[idx][0]
        duration = self.list[idx][0]            
        labels = self.list[idx][1]

                    
        result['video'] = inputs
        result['label'] = int(labels)
        result['duration'] = self.load_duration(duration.replace('.mp4', '.txt')).astype(np.bool)
        savename = self.list[idx][0].replace('F:\dataset\lrw_easy_to_err', target_dir).replace('.mp4', '.pkl')
        torch.save(result, savename)
        
        return result

    def __len__(self):
        return len(self.list)

    def load_duration(self, file):
        with open(file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if(line.find('Duration') != -1):
                    duration = float(line.split(' ')[1])
        
        tensor = np.zeros(29)
        mid = 29 / 2
        start = int(mid - duration / 2 * 25)
        end = int(mid + duration / 2 * 25)
        tensor[start:end] = 1.0
        return tensor            

if(__name__ == '__main__'):
    loader = DataLoader(LRWDataset(),
            batch_size = 96, 
            num_workers = 16,   
            shuffle = False,         
            drop_last = False)
    
    import time
    tic = time.time()
    for i, batch in enumerate(loader):
        toc = time.time()
        eta = ((toc - tic) / (i + 1) * (len(loader) - i)) / 3600.0
        print(f'eta:{eta:.5f}')        
