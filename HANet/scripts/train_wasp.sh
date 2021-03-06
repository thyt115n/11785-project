#!/usr/bin/env bash
    # Example on Cityscapes
     python -m torch.distributed.launch --nproc_per_node=4 train.py \
        --dataset cityscapes \
        --arch network.deepv3wasp.DeepR101V3PlusD_HANet \
        --city_mode 'train' \
        --lr_schedule poly \
        --lr 0.01 \
        --poly_exp 0.9 \
        --hanet_lr 0.01 \
        --hanet_poly_exp 0.9 \
        --max_cu_epoch 10000 \
        --class_uniform_pct 0.5 \
        --class_uniform_tile 1024 \
        --syncbn \
        --sgd \
        --crop_size 768 \
        --scale_min 0.5 \
        --scale_max 2.0 \
        --rrotate 0 \
        --color_aug 0.25 \
        --gblur \
        --max_iter 40000 \
        --bs_mult 4 \
        --hanet 1 1 1 1 0 \
        --hanet_set 3 32 3 \
        --hanet_pos 2 1 \
        --pos_rfactor 8 \
        --dropout 0.1 \
        --pos_noise 0.5 \
        --aux_loss \
        --date 0110 \
        --exp r101_os16_hanet \
        --ckpt ./logs/ \
        --tb_path ./logs/

