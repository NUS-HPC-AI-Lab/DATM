#!/bin/bash
export WANDB_API_KEY=cf58840c673fd8aa0611b91a7fc5f303665bdf24


CUDA_VISIBLE_DEVICES=0 python distill_FTD.py --dataset=CIFAR10 --ipc=10 --syn_steps=30 --expert_epochs=2 --max_start_epoch=20 --zca \
    --lr_img=100 --lr_lr=1e-05 --lr_teacher=0.001 --buffer_path=../buffer_storage/ --data_path=../dataset/ --ema_decay=0.9995 --Iteration=5000

CUDA_VISIBLE_DEVICES=0 python distill_FTD.py --dataset=CIFAR100 --ipc=10 --syn_steps=20 --expert_epochs=2 --max_start_epoch=40 --zca \
    --lr_img=1000 --lr_lr=1e-05 --lr_teacher=0.01 --buffer_path=../buffer_storage/ --data_path=../dataset/ --ema_decay=0.9995 --Iteration=5000

'''test, CIFAR100, 50ipc'''
python test.py --dataset=CIFAR100 --ipc=50 --syn_steps=80 --expert_epochs=2 --max_start_epoch=40 --zca \
    --lr_img=1000 --lr_lr=1e-05 --lr_teacher=0.01 --buffer_path=../buffer_storage/ --data_path=../dataset/ --ema_decay=0.999 --Iteration=10000 \
    --lr_y=2. --Momentum_y=0.9  --project=CIFAR100_ipc50 --threshold=1. --pix_init=samples_predicted_correctly --Initial_Step=20 \
    --learn_end_epoch=3000 --num_eval=1 --batch_syn=500 --eval_it=500

# ipc1000, test ResNet18
CUDA_VISIBLE_DEVICES=3,4,5 python distill_FTD.py --dataset=CIFAR10 --ipc=1000 --syn_steps=30 --expert_epochs=2 --max_start_epoch=40 --zca \
    --lr_img=1000 --lr_lr=1e-05 --lr_teacher=0.01 --buffer_path=../buffer_storage/CIFAR10_ipc50 --data_path=../dataset/ --ema_decay=0.999 --Iteration=5000 \
    --project=DD_Test --record_loss=True --num_eval=1 --batch_syn=1000 --eval_mode=M

# Tiny FTD ipc10
CUDA_VISIBLE_DEVICES=0,1 python distill_FTD.py --dataset=Tiny --ipc=10 --syn_steps=20 --expert_epochs=2 --max_start_epoch=40 --zca \
    --lr_img=10000 --lr_lr=1e-04 --lr_teacher=0.01 --buffer_path=../buffer_storage/a=1/ --data_path=../dataset/tiny-imagenet-200 --ema_decay=0.999 --Iteration=10000 \
    --project=Tiny_ipc10 --model=ConvNetD4 --num_eval=5 --batch_syn=500 --eval_it=500

# Ours FTD ipc10
CUDA_VISIBLE_DEVICES=2,3 python test.py --dataset=Tiny --ipc=10 --syn_steps=20 --expert_epochs=2 --max_start_epoch=40 --zca \
    --lr_img=10000 --lr_lr=1e-04 --lr_teacher=0.01 --buffer_path=../buffer_storage/a=1/ --data_path=../dataset/tiny-imagenet-200 --ema_decay=0.999 --Iteration=10000 \
    --project=Tiny_ipc10 --model=ConvNetD4 --num_eval=1 --batch_syn=500 --eval_it=500\
    --lr_y=2. --Momentum_y=0.9  --threshold=1. --pix_init=samples_predicted_correctly --Initial_Step=15 \
    --learn_end_epoch=3000
