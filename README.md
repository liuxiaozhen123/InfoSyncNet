# InfoSyncNet
The paper has been successfully published at the IJCNN 2025 conference, focusing on the visual speech recognition(VSR) task.
## How to train
### 1 Dependency installation
The required packages can be installed according to the instructions in `run.txt`.
### 2 Dataset Preprocessing 
- For the **LRW** dataset, run `python prepare_lrw.py` for preprocessing.  
- For the **LRW-1000** dataset, run `python prepare_lrw1000.py` for preprocessing.
### Pretrain Weights （optional）
We provide pretrained weight on LRW dataset for evaluation.
Link of pretrained weights: 
If you can not access to provided links, please email liuxiaozhen123@gs.zzu.edu.cn

### 3 Train
For example, to train the best model on the LRW:
```bash
python main_visual.py \
    --gpus='0' \
    --lr=3e-4 \
    --batch_size=32 \
    --num_workers=8 \
    --max_epoch=120 \
    --test=False \
    --save_prefix='checkpoints/lrw-best/' \
    --n_class=500 \
    --dataset='lrw' \
    --border=True \
    --mixup=True \
    --label_smooth=True \
    --se=True
```
to train the best model on the LRW1000:
```bash
python main_visual.py \
    --gpus='0' \
    --lr=3e-4 \
    --batch_size=32 \
    --num_workers=8 \
    --max_epoch=120 \
    --test=False \
    --save_prefix='checkpoints/lrw1000-best/' \
    --n_class=500 \
    --dataset='lrw1000' \
    --border=True \
    --mixup=True \
    --label_smooth=True \
    --se=True
```
### 4 Test
To test our provided weights, you should download weights and place them in the root of this repository.
For example, to test the best model on LRW Dataset:
```bash
python main_visual.py \
    --gpus='0' \
    --lr=3e-4 \
    --batch_size=32 \
    --num_workers=8 \
    --max_epoch=120 \
    --test=True \
    --save_prefix='checkpoints/lrw100-best/' \
    --n_class=500 \
    --dataset='lrw' \
    --border=True \
    --mixup=True \
    --label_smooth=True \
    --se=True
    --weights='checkpoints/lrw-cosine-lr-acc-0.85080.pt'
```














