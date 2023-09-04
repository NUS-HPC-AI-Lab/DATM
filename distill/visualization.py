import os
import sys
sys.path.append("../")
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
from tqdm import tqdm
from utils.utils_baseline import get_dataset, get_network, get_eval_pool, evaluate_synset, get_time, DiffAugment, ParamDiffAug, epoch
import wandb
import copy
import random
from reparam_module import ReparamModule
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

#!/usr/bin/env python
# _*_ coding:utf-8 _*_

 
 


def main(args):

    
    print("CUDNN STATUS: {}".format(torch.backends.cudnn.enabled))

    args.dsa = True if args.dsa == 'True' else False
    args.device ='cpu'


    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(args.dataset, args.data_path, args.batch_real, args.subset, args=args)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    im_res = im_size[0]

    args.im_size = im_size

    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    data_save = []

    if args.dsa:
        # args.epoch_eval_train = 1000
        args.dc_aug_param = None

    args.dsa_param = ParamDiffAug()

    dsa_params = args.dsa_param
    if args.zca:
        zca_trans = args.zca_trans
    else:
        zca_trans = None


    args.dsa_param = dsa_params
    args.zca_trans = zca_trans

    args.distributed = torch.cuda.device_count() > 1

    print('Hyper-parameters: \n', args.__dict__)
    print('Evaluation model pool: ', model_eval_pool)
    

    imgs = torch.load(args.distilled_dataset)
    imgs = args.zca_trans.inverse_transform(imgs.cpu())
    # imgs = torch.repeat_interleave(imgs, repeats=4, dim=2)
    # imgs = torch.repeat_interleave(imgs, repeats=4, dim=3)
    grid = torchvision.utils.make_grid(imgs[::100], nrow=10, normalize=True, scale_each=True)
    torchvision.utils.save_image(grid, 'test.jpg')
    # save_image_tensor2cv2(grid, 'test.jpg')
    print(imgs.shape)





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')

    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')

    parser.add_argument('--subset', type=str, default='imagenette', help='ImageNet subset. This only does anything when --dataset=ImageNet')

    parser.add_argument('--model', type=str, default='ConvNet', help='model')

    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')

    parser.add_argument('--eval_mode', type=str, default='S',
                        help='eval_mode, check utils.py for more info')

    parser.add_argument('--num_eval', type=int, default=5, help='how many networks to evaluate on')

    parser.add_argument('--eval_it', type=int, default=100, help='how often to evaluate')
    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=5000, help='how many distillation steps to perform')
    parser.add_argument('--lr_init', type=float, default=0.01, help='how to init lr (alpha)')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_syn', type=int, default=None, help='should only use this if you run out of VRAM')
    parser.add_argument('--batch_train', type=int, default=128, help='batch size for training networks')
    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                        help='whether to use differentiable Siamese augmentation.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--buffer_path', type=str, default='./buffers', help='buffer path')
    parser.add_argument('--zca', action='store_true', help="do ZCA whitening")
    parser.add_argument('--lr_teacher', type=float, default=0.01, help='initialization for synthetic learning rate')
    parser.add_argument('--no_aug', type=bool, default=False, help='this turns off diff aug during distillation')
    parser.add_argument('--data_dir', type=str, default='path', help='dataset')
    parser.add_argument('--label_dir', type=str, default='path', help='dataset')
    parser.add_argument('--lr_dir', type=str, default='path', help='dataset')
    parser.add_argument('--parall_eva', type=bool, default=False, help='dataset')
    parser.add_argument('--ASL_model', type=str, default=None)
    parser.add_argument('--ASL_model_dir', type=str, default=None)
    parser.add_argument('--method', type=str, default='')
    parser.add_argument('--distilled_dataset', type=str, default='/home/guoziyao/Gzy/FTD-distillation/distill/logged_files/CIFAR10/1000/ConvNet/winter-rain-91/Normal/images_10000.pt')

    args = parser.parse_args()
    main(args)

'''
sudo CUDA_VISIBLE_DEVICES=3 python /cpfs01/shared/public/Gzy/FTD-distillation/distill/visualization.py \
--dataset=CIFAR10 \
--data_path=/cpfs01/shared/public/Gzy/FTD-distillation/dataset \
--zca \


--ASL_model=ConvNet \
--ASL_model_dir=/cpfs01/shared/public/Gzy/FTD-distillation/buffer_storage/CIFAR100/ConvNet




sudo CUDA_VISIBLE_DEVICES=1
python /cpfs01/shared/public/Gzy/FTD-distillation/distill/evaluation.py \
--dataset=CIFAR10 \
--model=ConvNet \
--data_path=/cpfs01/shared/public/Gzy/FTD-distillation/dataset \
--lr_dir=/cpfs01/shared/public/Gzy/Distilled_Datasets/FTD/CIFAR-10/IPC1/Normal/lr_best.pt \
--data_dir=/cpfs01/shared/public/Gzy/Distilled_Datasets/FTD/CIFAR-10/IPC1/Normal/images_best.pt \
--label_dir=/cpfs01/shared/public/Gzy/Distilled_Datasets/FTD/CIFAR-10/IPC1/Normal/labels_best.pt \
--eval_mode=Maa \
--subset=imagenette \
--zca \
--ASL_model=ConvNet \
--ASL_model_dir=/cpfs01/shared/public/Gzy/FTD-distillation/buffer_storage/CIFAR10_ipc50/CIFAR10/ConvNet



sudo CUDA_VISIBLE_DEVICES=0 python /cpfs01/shared/public/Gzy/FTD-distillation/distill/evaluation.py \
--dataset=CIFAR100 \
--zca \
--model=ConvNet \
--data_path=/home/guoziyao/Gzy/FTD-distillation/dataset \
--method=Random \
--ipc=50 \
--eval_mode=M
'''
