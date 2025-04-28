# SCM-ReID
The official repository for Occluded Person Re-identification with Spectral Clustering Mask.

## Prepare Datasets
Download the person datasets, vehicle datasets, and fine-grained Visual Categorization/Retrieval datasets.

Then unzip them and rename them under your "dataset_root" directory like
```bash
dataset_root
├── Occluded_Duke
├── P-DukeMTMC-reid
├── Market-1501-v15.09.15
├── DukeMTMC-reID
├── MSMT17
├── cuhk03-np
├── VeRi
├── VehicleID_V1.0
├── CARS
├── CUB_200_2011
└── University-Release
```

## Training
We prepared the ImageNet Pretrained RegNet backbone in "./pretrain".

### Train on Occluded_Duke
```bash
python train.py --net regnet_y_800mf --img-height 384 --img-width 128 --batch-size 24 --lr 4.0e-2 --dataset occluded_duke --gpus 0 --epochs 5,75 --instance-num 4 --erasing 0.40 --mask-part 1 --kernel-size 4 --num-part 3 --triplet-weight 1.0 --mask-weight 3e-3 --feat-num 256 --reduce-num 32 --freeze stem --dataset-root ../datasets --ema-ratio 0.80 --ema-extra 25
```
top1:0.705430 top5:0.831222 top10:0.873756 mAP:0.596226
```bash
python train.py --net regnet_y_1_6gf --img-height 384 --img-width 128 --batch-size 24 --lr 4.0e-2 --dataset occluded_duke --gpus 0 --epochs 5,75 --instance-num 4 --erasing 0.40 --mask-part 1 --kernel-size 4 --num-part 3 --triplet-weight 1.0 --mask-weight 3e-3 --feat-num 256 --reduce-num 32 --freeze stem --dataset-root ../datasets --ema-ratio 0.80 --ema-extra 25
```
top1:0.722624 top5:0.837557 top10:0.876018 mAP:0.615356

### Train on P-DukeMTMC
```bash
python train.py --net regnet_y_800mf --img-height 384 --img-width 128 --batch-size 24 --lr 4.0e-2 --dataset p_dukemtmc --gpus 0 --epochs 5,75 --instance-num 4 --erasing 0.40 --mask-part 1 --kernel-size 4 --num-part 3 --triplet-weight 1.0 --mask-weight 3e-3 --feat-num 256 --reduce-num 32 --freeze stem --dataset-root ../datasets --ema-ratio 0.80 --ema-extra 25
```
top1:0.923255 top5:0.950994 top10:0.963477 mAP:0.829040
```bash
python train.py --net regnet_y_1_6gf --img-height 384 --img-width 128 --batch-size 24 --lr 4.0e-2 --dataset p_dukemtmc --gpus 0 --epochs 5,75 --instance-num 4 --erasing 0.40 --mask-part 1 --kernel-size 4 --num-part 3 --triplet-weight 1.0 --mask-weight 3e-3 --feat-num 256 --reduce-num 32 --freeze stem --dataset-root ../datasets --ema-ratio 0.80 --ema-extra 25
```
top1:0.925566 top5:0.955155 top10:0.962552 mAP:0.833821

### Train on Market1501
```bash
python train.py --net regnet_y_800mf --img-height 384 --img-width 128 --batch-size 24 --lr 4.0e-2 --dataset market1501 --gpus 0 --epochs 5,75 --instance-num 4 --erasing 0.40 --mask-part 1 --kernel-size 4 --num-part 3 --triplet-weight 1.0 --mask-weight 3e-3 --feat-num 256 --reduce-num 32 --freeze stem --dataset-root ../datasets --ema-ratio 0.80 --ema-extra 25
```
top1:0.958729 top5:0.985748 top10:0.990796 mAP:0.897846
```bash
python train.py --net regnet_y_1_6gf --img-height 384 --img-width 128 --batch-size 24 --lr 4.0e-2 --dataset market1501 --gpus 0 --epochs 5,75 --instance-num 4 --erasing 0.40 --mask-part 1 --kernel-size 4 --num-part 3 --triplet-weight 1.0 --mask-weight 3e-3 --feat-num 256 --reduce-num 32 --freeze stem --dataset-root ../datasets --ema-ratio 0.80 --ema-extra 25
```
top1:0.960808 top5:0.984264 top10:0.991686 mAP:0.899680

### Train on DukeMTMC
```bash
python train.py --net regnet_y_800mf --img-height 384 --img-width 128 --batch-size 24 --lr 4.0e-2 --dataset dukemtmc --gpus 0 --epochs 5,75 --instance-num 4 --erasing 0.40 --mask-part 1 --kernel-size 4 --num-part 3 --triplet-weight 1.0 --mask-weight 3e-3 --feat-num 256 --reduce-num 32 --freeze stem --dataset-root ../datasets --ema-ratio 0.80 --ema-extra 25
```
top1:0.913375 top5:0.954668 top10:0.965889 mAP:0.816747
```bash
python train.py --net regnet_y_1_6gf --img-height 384 --img-width 128 --batch-size 24 --lr 4.0e-2 --dataset dukemtmc --gpus 0 --epochs 5,75 --instance-num 4 --erasing 0.40 --mask-part 1 --kernel-size 4 --num-part 3 --triplet-weight 1.0 --mask-weight 3e-3 --feat-num 256 --reduce-num 32 --freeze stem --dataset-root ../datasets --ema-ratio 0.80 --ema-extra 25
```
top1:0.912926 top5:0.956463 top10:0.968133 mAP:0.824236

### Train on CUHK03 Detected
```bash
python train.py --net regnet_y_800mf --img-height 384 --img-width 128 --batch-size 24 --lr 4.0e-2 --dataset npdetected --gpus 0 --epochs 5,155 --instance-num 4 --erasing 0.40 --mask-part 1 --kernel-size 4 --num-part 3 --triplet-weight 1.0 --mask-weight 3e-3 --feat-num 256 --reduce-num 32 --freeze stem --dataset-root ../datasets --ema-ratio 0.80 --ema-extra 25
```
top1:0.822143 top5:0.927143 top10:0.964286 mAP:0.787892
```bash
python train.py --net regnet_y_1_6gf --img-height 384 --img-width 128 --batch-size 24 --lr 4.0e-2 --dataset npdetected --gpus 0 --epochs 5,155 --instance-num 4 --erasing 0.40 --mask-part 1 --kernel-size 4 --num-part 3 --triplet-weight 1.0 --mask-weight 3e-3 --feat-num 256 --reduce-num 32 --freeze stem --dataset-root ../datasets --ema-ratio 0.80 --ema-extra 25
```
top1:0.827143 top5:0.922143 top10:0.960000 mAP:0.788114

### Train on CUHK03 Labeled
```bash
python train.py --net regnet_y_800mf --img-height 384 --img-width 128 --batch-size 24 --lr 4.0e-2 --dataset nplabeled --gpus 0 --epochs 5,155 --instance-num 4 --erasing 0.40 --mask-part 1 --kernel-size 4 --num-part 3 --triplet-weight 1.0 --mask-weight 3e-3 --feat-num 256 --reduce-num 32 --freeze stem --dataset-root ../datasets --ema-ratio 0.80 --ema-extra 25
```
top1:0.846429 top5:0.942857 top10:0.967857 mAP:0.817966
```bash
python train.py --net regnet_y_1_6gf --img-height 384 --img-width 128 --batch-size 24 --lr 4.0e-2 --dataset nplabeled --gpus 0 --epochs 5,155 --instance-num 4 --erasing 0.40 --mask-part 1 --kernel-size 4 --num-part 3 --triplet-weight 1.0 --mask-weight 3e-3 --feat-num 256 --reduce-num 32 --freeze stem --dataset-root ../datasets --ema-ratio 0.80 --ema-extra 25
```
top1:0.850714 top5:0.936429 top10:0.967143 mAP:0.820306

### Train on MSMT17
```bash
python train.py --net regnet_y_800mf --img-height 384 --img-width 128 --batch-size 24 --lr 4.0e-2 --dataset msmt17 --gpus 0 --epochs 5,75 --instance-num 4 --erasing 0.40 --mask-part 1 --kernel-size 4 --num-part 3 --triplet-weight 1.0 --mask-weight 3e-3 --feat-num 256 --reduce-num 32 --freeze stem --dataset-root ../datasets --ema-ratio 0.80 --ema-extra 25
```
top1:0.852389 top5:0.925723 top10:0.944249 mAP:0.654714
```bash
python train.py --net regnet_y_1_6gf --img-height 384 --img-width 128 --batch-size 24 --lr 4.0e-2 --dataset msmt17 --gpus 0 --epochs 5,75 --instance-num 4 --erasing 0.40 --mask-part 1 --kernel-size 4 --num-part 3 --triplet-weight 1.0 --mask-weight 3e-3 --feat-num 256 --reduce-num 32 --freeze stem --dataset-root ../datasets --ema-ratio 0.80 --ema-extra 25
```
top1:0.864825 top5:0.932842 top10:0.950167 mAP:0.677661

### Train on VeRI776
```bash
python train.py --net regnet_y_800mf --img-height 256 --img-width 256 --batch-size 24 --lr 4.0e-2 --dataset veri776 --gpus 0 --epochs 5,75 --instance-num 4 --erasing 0.40 --mask-part 1 --kernel-size 4 --num-part 3 --triplet-weight 1.0 --mask-weight 3e-3 --feat-num 256 --reduce-num 32 --freeze stem --dataset-root ../datasets --ema-ratio 0.80 --ema-extra 25
```
top1:0.976162 top5:0.984505 top10:0.988081 mAP:0.824434
```bash
python train.py --net regnet_y_1_6gf --img-height 256 --img-width 256 --batch-size 24 --lr 4.0e-2 --dataset veri776 --gpus 0 --epochs 5,75 --instance-num 4 --erasing 0.40 --mask-part 1 --kernel-size 4 --num-part 3 --triplet-weight 1.0 --mask-weight 3e-3 --feat-num 256 --reduce-num 32 --freeze stem --dataset-root ../datasets --ema-ratio 0.80 --ema-extra 25
```
top1:0.970203 top5:0.986889 top10:0.989273 mAP:0.822843

### Train on VehicleID
```bash
python train.py --net regnet_y_800mf --img-height 256 --img-width 256 --batch-size 256 --lr 2.0e-1 --dataset vehicleid --gpus 0 --epochs 5,75 --instance-num 4 --erasing 0.40 --mask-part 0 --kernel-size 4 --num-part 3 --triplet-weight 1.0 --mask-weight 1e-3 --feat-num 256 --reduce-num 32 --freeze stem --dataset-root ../datasets --ema-ratio 0.80 --ema-extra 25
```
top1:0.848937 top5:0.980908 top10:0.992271 mAP:0.876849
```bash
python train.py --net regnet_y_1_6gf --img-height 256 --img-width 256 --batch-size 256 --lr 2.0e-1 --dataset vehicleid --gpus 3 --epochs 5,75 --instance-num 4 --erasing 0.40 --mask-part 0 --kernel-size 4 --num-part 3 --triplet-weight 1.0 --mask-weight 1e-3 --feat-num 256 --reduce-num 32 --freeze stem --dataset-root ../datasets --ema-ratio 0.80 --ema-extra 25
```
top1:0.856139 top5:0.982259 top10:0.995784 mAP:0.883299

### Train on CUB200
```bash
python train.py --net regnet_y_800mf --img-height 256 --img-width 256 --batch-size 24 --lr 1.0e-3 --dataset cub200 --gpus 0 --epochs 5,45 --instance-num 6 --erasing 0.10 --mask-part 1 --kernel-size 4 --num-part 3 --triplet-weight 1.0 --mask-weight 3e-3 --feat-num 256 --reduce-num 32 --freeze stem --dataset-root ../datasets --ema-ratio 0.80 --ema-extra 25
```
Recall@1:0.717421 Recall@2:0.807562 Recall@4:0.876266 Recall@8:0.919649 NMI:0.724564
```bash
python train.py --net regnet_y_1_6gf --img-height 256 --img-width 256 --batch-size 24 --lr 1.0e-3 --dataset cub200 --gpus 0 --epochs 5,45 --instance-num 6 --erasing 0.10 --mask-part 1 --kernel-size 4 --num-part 3 --triplet-weight 1.0 --mask-weight 3e-3 --feat-num 256 --reduce-num 32 --freeze stem --dataset-root ../datasets --ema-ratio 0.80 --ema-extra 25
```
Recall@1:0.730250 Recall@2:0.820898 Recall@4:0.882849 Recall@8:0.927752 NMI:0.737068

### Train on Car196
```bash
python train.py --net regnet_y_800mf --img-height 256 --img-width 256 --batch-size 24 --lr 4.0e-2 --dataset car196 --gpus 0 --epochs 5,45 --instance-num 6 --erasing 0.10 --mask-part 1 --kernel-size 4 --num-part 3 --triplet-weight 1.0 --mask-weight 3e-3 --feat-num 256 --reduce-num 32 --freeze stem --dataset-root ../datasets --ema-ratio 0.80 --ema-extra 25
```
Recall@1:0.928668 Recall@2:0.956955 Recall@4:0.970852 Recall@8:0.980691 NMI:0.795589
```bash
python train.py --net regnet_y_1_6gf --img-height 256 --img-width 256 --batch-size 24 --lr 4.0e-2 --dataset car196 --gpus 0 --epochs 5,45 --instance-num 6 --erasing 0.10 --mask-part 1 --kernel-size 4 --num-part 3 --triplet-weight 1.0 --mask-weight 3e-3 --feat-num 256 --reduce-num 32 --freeze stem --dataset-root ../datasets --ema-ratio 0.80 --ema-extra 25
```
Recall@1:0.927684 Recall@2:0.958185 Recall@4:0.969499 Recall@8:0.980937 NMI:0.803123

### Train on University1652
```bash
python train.py --net regnet_y_800mf --img-height 256 --img-width 256 --batch-size 24 --lr 4.0e-2 --dataset university1652 --gpus 0 --epochs 3,15 --instance-num 6 --erasing 0.10 --mask-part 1 --kernel-size 4 --num-part 3 --triplet-weight 1.0 --mask-weight 3e-3 --feat-num 256 --reduce-num 32 --freeze stem --dataset-root ../datasets --ema-ratio 0.80 --ema-extra 15
```
top1:0.915835 top5:0.941512 top10:0.948645 mAP:0.895134
```bash
python train.py --net regnet_y_1_6gf --img-height 256 --img-width 256 --batch-size 24 --lr 4.0e-2 --dataset university1652 --gpus 0 --epochs 3,15 --instance-num 6 --erasing 0.10 --mask-part 1 --kernel-size 4 --num-part 3 --triplet-weight 1.0 --mask-weight 3e-3 --feat-num 256 --reduce-num 32 --freeze stem --dataset-root ../datasets --ema-ratio 0.80 --ema-extra 15
```
top1:0.925820 top5:0.951498 top10:0.957204 mAP:0.902602

## Contact
If you have any questions, please contact us by email(laishenqi@qq.com).