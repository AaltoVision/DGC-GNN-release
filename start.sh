#!/bin/bash
#SBATCH --job-name=dgc_gnn
#SBATCH --account=project_2003267
#SBATCH --partition=gpusmall
#SBATCH --time=32:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --gres=gpu:a100:1




python -m dgc_gnn_train.train_matcher --gpus 0 --batch 8 -lr 0.001 \
    --max_epochs 50 --matcher_class 'OTMatcherCls' --share_kp2d_enc --normalized_thres \
    --dataset 'megadepth' --train_split 'train' --val_split 'val' \
    --outlier_rate 0.5 0.5  --topk 1 --npts 100 1024 \
    --p2d_type 'sift' --p3d_type 'bvs' \
    --inls2d_thres 0.001 --rpthres 0.01 --prefix 'gomatchbvs' 



