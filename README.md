# InfoSyncNet
The paper has been successfully published at the IJCNN 2025 conference, focusing on the visual speech recognition(VSR) task.
## How to train
### 1 Dependency installation
The required packages can be installed according to the instructions in `run.txt`.
### 2 Dataset Preprocessing 
- For the **LRW** dataset, run `python prepare_lrw.py` for preprocessing.  
- For the **LRW-1000** dataset, run `python prepare_lrw1000.py` for preprocessing.

### 3 Train
For example, to train lrw baseline:
```bash
python main_visual.py \
    --gpus='0' \
    --lr=3e-4 \
    --batch_size=32 \
    --num_workers=8 \
    --max_epoch=120 \
    --test=False \
    --save_prefix='checkpoints/lrw-baseline/' \
    --n_class=500 \
    --dataset='lrw' \
    --border=True \
    --mixup=True \
    --label_smooth=True \
    --se=True
```
### 4 Test
