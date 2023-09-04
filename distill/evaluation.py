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



def main(args):

    
    print("CUDNN STATUS: {}".format(torch.backends.cudnn.enabled))

    args.dsa = True if args.dsa == 'True' else False
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'


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
    

    # criterion = nn.CrossEntropyLoss().to(args.device)
    def SoftCrossEntropy(inputs, target, reduction='average'):
        input_log_likelihood = -F.log_softmax(inputs, dim=1)
        target_log_likelihood = F.softmax(target, dim=1)
        batch = inputs.shape[0]
        loss = torch.sum(torch.mul(input_log_likelihood, target_log_likelihood)) / batch
        return loss
    def generate_soft_label(expert_files, image_syn, hard_label):
        logits=[]
        Temp_net = get_network(args.ASL_model, channel, num_classes, im_size, dist=False).to(args.device)  # get a random model
        Temp_net = ReparamModule(Temp_net)
        if args.distributed:
            Temp_net = torch.nn.DataParallel(Temp_net)
        Temp_net.eval()
        for i in range(len(expert_files)):
            Temp_Buffer = torch.load(expert_files[i])
            for j in Temp_Buffer:
                Temp_params = j[-1]
                Initialize_Labels_params = torch.cat([p.data.to(args.device).reshape(-1) for p in Temp_params], 0)
                if args.distributed:
                    Initialize_Labels_params = Initialize_Labels_params.unsqueeze(0).expand(torch.cuda.device_count(), -1)
                Initialized_Labels = Temp_net(image_syn.to(args.device), flat_param=Initialize_Labels_params)
                logits.append(Initialized_Labels.detach().cpu())
        print('use {} models to initialize soft label'.format(len(logits)))
        logits_tensor = torch.stack(logits)
        if len(hard_label.shape) !=1:
            true_labels = torch.argmax(hard_label, dim=1).cpu()
        else:
            true_labels = hard_label.cpu()
        predicted_labels = torch.argmax(logits_tensor, dim=2).cpu()
        correct_predictions = predicted_labels == true_labels.view(1, -1)
        mask = correct_predictions.unsqueeze(2)
        correct_logits = logits_tensor * mask.float()
        correct_logits_per_model = correct_logits.sum(dim=0)
        num_correct_images_per_model = correct_predictions.sum(dim=0, dtype=torch.float)
        average_logits_per_image = correct_logits_per_model / num_correct_images_per_model.unsqueeze(1) 
        Initialized_Labels = average_logits_per_image
        acc = np.sum(np.equal(np.argmax(Initialized_Labels.cpu().data.numpy(), axis=-1), true_labels.cpu().data.numpy()))
        print('InitialAcc:{}'.format(acc/len(true_labels)))
        if acc/len(true_labels)<1.:
            print(Initialized_Labels.shape)
            Initialized_Labels = torch.sum(logits_tensor,dim=0)/len(logits_tensor)
        return Initialized_Labels.detach()
    criterion = nn.CrossEntropyLoss().to(args.device)
    soft_cri = SoftCrossEntropy


    if args.method == 'Random':
        print('-----Random Selection-------')
        ''' organize the real dataset '''
        images_all = []
        labels_all = []
        indices_class = [[] for c in range(num_classes)]
        print("BUILDING DATASET")
        for i in tqdm(range(len(dst_train))):
            sample = dst_train[i]
            images_all.append(torch.unsqueeze(sample[0], dim=0))
            labels_all.append(class_map[torch.tensor(sample[1]).item()])
        for i, lab in tqdm(enumerate(labels_all)):
            indices_class[lab].append(i)
        images_all = torch.cat(images_all, dim=0).to("cpu")
        labels_all = torch.tensor(labels_all, dtype=torch.long, device="cpu")
        def get_images(c, n):  # get random n images from class c
            idx_shuffle = np.random.permutation(indices_class[c])[:n]
            return images_all[idx_shuffle]
        label_syn_eval = torch.tensor([np.ones(args.ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]
        image_syn_eval = torch.randn(size=(num_classes * args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float)
        args.lr_net = torch.tensor(args.lr_teacher).to(args.device)
        for c in range(num_classes):
            image_syn_eval.data[c * args.ipc:(c + 1) * args.ipc] = get_images(c, args.ipc).detach().data

    else:
        # test
        image_syn_eval = torch.load(args.data_dir)
        label_syn_eval = torch.load(args.label_dir)
        args.lr_net = torch.load(args.lr_dir)
        if args.ASL_model is not None:
            expert_files = []
            n = 0
            expert_dir = args.ASL_model_dir
            while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
                expert_files.append(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
                n += 1
            if n == 0:
                raise AssertionError("No buffers detected at {}".format(expert_dir))

    for model_eval in model_eval_pool:
        print('Evaluating: '+model_eval)
        network = get_network(model_eval, channel, num_classes, im_size, dist=False).to(args.device)  # get a random model
        
        # for i in network.parameters():
        #     print(len(i))
        # print(network.parameters)
        # for name in network.state_dict():
        #     print(name)

        # weights_list = torch.load('/home/guoziyao/Gzy/FTD-distillation/buffer_storage/ipc50/CIFAR10/ConvNetBN/replay_buffer_0.pt')[-2][-1]
        # # Iterate through the model's parameters and assign weights from the list
        # for param, loaded_weight in zip(network.parameters(), weights_list):
        #     param.data = loaded_weight
        # args.epoch_eval_train = 0
        # test_criterion = nn.CrossEntropyLoss().to(args.device)
        # loss_test, acc_test = epoch('test', testloader, network, None, test_criterion, args, aug=False)
        # print(acc_test)


        if len(label_syn_eval.shape)!=1:
            _, acc_train, acc_test = evaluate_synset(0, copy.deepcopy(network), image_syn_eval, label_syn_eval, testloader, args, texture=False, train_criterion=soft_cri)
        else:
            _, acc_train, acc_test = evaluate_synset(0, copy.deepcopy(network), image_syn_eval, label_syn_eval, testloader, args, texture=False)
        if args.ASL_model is not None:
            assigned_soft_label=generate_soft_label(expert_files, image_syn_eval, label_syn_eval)
            _, _, acc_test_soft_label = evaluate_synset(0, copy.deepcopy(network), image_syn_eval, assigned_soft_label, testloader, args, texture=False, train_criterion=soft_cri)



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

    args = parser.parse_args()
    main(args)

'''
sudo CUDA_VISIBLE_DEVICES=2 python /cpfs01/shared/public/Gzy/FTD-distillation/distill/evaluation.py \
--dataset=Tiny \
--model=ConvNetD4 \
--data_path=/cpfs01/shared/public/Gzy/FTD-distillation/dataset/tiny-imagenet-200 \
--lr_dir=/cpfs01/shared/public/Gzy/Distilled_Datasets/Ours/Tiny/IPC50/divine-violet-36/Normal/lr_best.pt \
--data_dir=/cpfs01/shared/public/Gzy/Distilled_Datasets/Ours/Tiny/IPC50/divine-violet-36/Normal/images_best.pt \
--label_dir=/cpfs01/shared/public/Gzy/Distilled_Datasets/Ours/Tiny/IPC50/divine-violet-36/Normal/labels_best.pt \
--eval_mode=Maa \
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
