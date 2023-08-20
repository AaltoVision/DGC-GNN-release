#!/bin/bash


python -m dgc_gnn_eval.benchmark  --root .  --ckpt 'pretrained/best.ckpt' \
    --splits 'test'  \
    --odir 'outputs/benchmark_cache_release' \
    --dataset 'megadepth' --covis_k_nums 1  \
    --p2d_type 'sift' \
    --merge_before_match 