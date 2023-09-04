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
from utils.utils_baseline import get_dataset, get_network, get_eval_pool, evaluate_synset, get_time, DiffAugment, ParamDiffAug
import wandb
import copy
import random
from reparam_module import ReparamModule
from model_ema import ModelEmaV2

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def main(args):

    if args.zca and args.texture:
        raise AssertionError("Cannot use zca and texture together")

    if args.texture and args.pix_init == "real":
        print("WARNING: Using texture with real initialization will take a very long time to smooth out the boundaries between images.")

    if args.max_experts is not None and args.max_files is not None:
        args.total_experts = args.max_experts * args.max_files

    print("CUDNN STATUS: {}".format(torch.backends.cudnn.enabled))

    args.dsa = True if args.dsa == 'True' else False
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.skip_first_eva==False:
        eval_it_pool = np.arange(0, args.Iteration + 1, args.eval_it).tolist()
    else:
        eval_it_pool = np.arange(args.eval_it, args.Iteration + 1, args.eval_it).tolist()

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

    wandb.init(sync_tensorboard=False,
               project=args.project,
               job_type="CleanRepo",
               config=args,
               )

    args = type('', (), {})()

    for key in wandb.config._items:
        setattr(args, key, wandb.config._items[key])

    args.dsa_param = dsa_params
    args.zca_trans = zca_trans

    if args.batch_syn is None:
        args.batch_syn = num_classes * args.ipc

    args.distributed = torch.cuda.device_count() > 1


    print('Hyper-parameters: \n', args.__dict__)
    print('Evaluation model pool: ', model_eval_pool)

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

    for c in range(num_classes):
        print('class c = %d: %d real images'%(c, len(indices_class[c])))

    for ch in range(channel):
        print('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))


    def get_images(c, n):  # get random n images from class c
        idx_shuffle = np.random.permutation(indices_class[c])[:n]
        return images_all[idx_shuffle]


    ''' initialize the synthetic data '''
    label_syn = torch.tensor([np.ones(args.ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]

    if args.texture:
        image_syn = torch.randn(size=(num_classes * args.ipc, channel, im_size[0]*args.canvas_size, im_size[1]*args.canvas_size), dtype=torch.float)
    else:
        image_syn = torch.randn(size=(num_classes * args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float)

    syn_lr = torch.tensor(args.lr_teacher).to(args.device)

    criterion = nn.CrossEntropyLoss().to(args.device)
    print('%s training begins'%get_time())

    expert_dir = os.path.join(args.buffer_path, args.dataset)
    if args.dataset == "ImageNet":
        expert_dir = os.path.join(expert_dir, args.subset, str(args.res))
    if args.dataset in ["CIFAR10", "CIFAR100"] and not args.zca:
        expert_dir += "_NO_ZCA"
    expert_dir = os.path.join(expert_dir, args.model)
    print("Expert Dir: {}".format(expert_dir))

    if args.load_all:
        buffer = []
        n = 0
        while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
            buffer = buffer + torch.load(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
            n += 1
        if n == 0:
            raise AssertionError("No buffers detected at {}".format(expert_dir))

    else:
        expert_files = []
        n = 0
        while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
            expert_files.append(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
            n += 1
        if n == 0:
            raise AssertionError("No buffers detected at {}".format(expert_dir))
        file_idx = 0
        expert_idx = 0
        # random.shuffle(expert_files)
        if args.max_files is not None:
            expert_files = expert_files[:args.max_files]
        
        expert_id = [i for i in range(len(expert_files))]
        random.shuffle(expert_id)

        print("loading file {}".format(expert_files[expert_id[file_idx]]))
        buffer = torch.load(expert_files[expert_id[file_idx]])
        if args.max_experts is not None:
            buffer = buffer[:args.max_experts]
        buffer_id = [i for i in range(len(buffer))]
        random.shuffle(buffer_id)



    if args.pix_init == 'real':
        print('initialize synthetic data from random real images')
        if args.texture:
            for c in range(num_classes):
                for i in range(args.canvas_size):
                    for j in range(args.canvas_size):
                        image_syn.data[c * args.ipc:(c + 1) * args.ipc, :, i * im_size[0]:(i + 1) * im_size[0],
                        j * im_size[1]:(j + 1) * im_size[1]] = torch.cat(
                            [get_images(c, 1).detach().data for s in range(args.ipc)])
        for c in range(num_classes):
            image_syn.data[c * args.ipc:(c + 1) * args.ipc] = get_images(c, args.ipc).detach().data
    

        # Temp_net = get_network(args.model, channel, num_classes, im_size, dist=False).to(args.device)  # get a random model
        # Temp_net = ReparamModule(Temp_net)
        # if args.distributed:
        #     Temp_net = torch.nn.DataParallel(Temp_net)
        # Temp_net.eval()
        # logits=[]
        # cur=0
        # temp_params = torch.load(expert_files[0])[0][-1]
        # temp_params = torch.cat([p.data.to(args.device).reshape(-1) for p in temp_params], 0)
        # if args.distributed:
        #     temp_params = temp_params.unsqueeze(0).expand(torch.cuda.device_count(), -1)
        # for c in range(num_classes):
        #     data_for_class_c = get_images(c, len(indices_class[c])).detach().data
        #     n, _, w, h = data_for_class_c.shape
        #     selected_num = 0
        #     select_times = 0
        #     temp_img = None
        #     Wrong_Predicted_Img = None
        #     cur=0
        #     batch_size = 256
        #     index = []
        #     while len(index)<args.ipc:
        #         current_data_batch = data_for_class_c[batch_size*select_times : batch_size*(select_times+1)].detach().to(args.device)
        #         print(str(c)+'.'+str(cur))
        #         if batch_size*select_times > len(data_for_class_c):
        #             select_times = 0
        #             cur+=1
        #             temp_params = torch.load(expert_files[int(cur/10)])[cur%10][-1]
        #             temp_params = torch.cat([p.data.to(args.device).reshape(-1) for p in temp_params], 0)
        #             continue
        #         logits = Temp_net(current_data_batch, flat_param=temp_params).detach()
        #         prediction_class = np.argmax(logits.cpu().data.numpy(), axis=-1)
        #         for i in range(len(prediction_class)):
        #             if prediction_class[i]==c and len(index)<args.ipc:
        #                 if batch_size*select_times+i not in index:
        #                     index.append(batch_size*select_times+i)
        #         select_times+=1
        #         if len(index) == args.ipc:
        #             temp_img = torch.index_select(data_for_class_c, dim=0, index=torch.tensor(index))
        #             break
        #     image_syn.data[c * args.ipc:(c + 1) * args.ipc] = temp_img.detach()
    
    
    else:
        print('initialize synthetic data from random noise')


    ''' training '''
    image_syn = image_syn.detach().to(args.device).requires_grad_(True)
    syn_lr = syn_lr.detach().to(args.device).requires_grad_(True)



    ###EMA modification
    
    optimizer_img = torch.optim.SGD([image_syn], lr=args.lr_img, momentum=0.5)
    optimizer_lr = torch.optim.SGD([syn_lr], lr=args.lr_lr, momentum=0.5)
    ema = ModelEmaV2([image_syn], decay=args.ema_decay)
    optimizer_img.zero_grad()

    ###


    
    best_acc = {m: 0 for m in model_eval_pool}
    best_std = {m: 0 for m in model_eval_pool}
    

    ''' Evaluate ema synthetic data '''
    ema_best_acc = {m: 0 for m in model_eval_pool}
    ema_best_std = {m: 0 for m in model_eval_pool}

    expert_step_store = {}


    def generate_soft_label(expert_files, image_syn, hard_label):
        logits=[]
        Temp_net = get_network(args.model, channel, num_classes, im_size, dist=False).to(args.device)  # get a random model
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
                Initialized_Labels = Temp_net(image_syn, flat_param=Initialize_Labels_params)
                logits.append(Initialized_Labels.detach().cpu())
        print('use {} models to initialize soft label'.format(len(logits)))
        logits_tensor = torch.stack(logits)
        true_labels = hard_label.cpu()
        predicted_labels = torch.argmax(logits_tensor, dim=2).cpu()
        correct_predictions = predicted_labels == true_labels.view(1, -1)
        mask = correct_predictions.unsqueeze(2)
        correct_logits = logits_tensor * mask.float()
        correct_logits_per_model = correct_logits.sum(dim=0)
        num_correct_images_per_model = correct_predictions.sum(dim=0, dtype=torch.float)
        average_logits_per_image = correct_logits_per_model / num_correct_images_per_model.unsqueeze(1) 
        Initialized_Labels = average_logits_per_image
        acc = np.sum(np.equal(np.argmax(Initialized_Labels.cpu().data.numpy(), axis=-1), hard_label.cpu().data.numpy()))
        print('InitialAcc:{}'.format(acc/len(hard_label)))
        return Initialized_Labels.detach()
    def SoftCrossEntropy(inputs, target, reduction='average'):
        input_log_likelihood = -F.log_softmax(inputs, dim=1)
        target_log_likelihood = F.softmax(target, dim=1)
        batch = inputs.shape[0]
        loss = torch.sum(torch.mul(input_log_likelihood, target_log_likelihood)) / batch
        # loss = F.kl_div(input_log_likelihood, target_log_likelihood, reduction="none").sum(1).mean()
        return loss



    for it in range(0, args.Iteration+1):
        save_this_it = False

        # writer.add_scalar('Progress', it, it)
        wandb.log({"Progress": it}, step=it)
        ''' Evaluate synthetic data '''
        if it in eval_it_pool:
            
            # test
            if args.record_loss:
                
                student_net = get_network(args.model, channel, num_classes, im_size, dist=False).to(args.device)  # get a random model
                student_net = ReparamModule(student_net)

                if args.distributed and args.model[-2:]!='BN':
                    student_net = torch.nn.DataParallel(student_net)

                student_net.eval()
                num_params = sum([np.prod(p.size()) for p in (student_net.parameters())])
                buffer = torch.load(expert_files[-1])

                log_length = 45
                log_dict = {}

                for i in range(log_length):
                    starting_params = buffer[-1][i]
                    target_params = buffer[-1][i+args.expert_epochs]
                    target_params = torch.cat([p.data.to(args.device).reshape(-1) for p in target_params], 0)
                    student_params = [torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0).requires_grad_(True)]
                    starting_params = torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0)
                    syn_images = image_syn
                    y_hat = label_syn.to(args.device)
                    param_loss_list = []
                    param_dist_list = []
                    indices_chunks = []
                    for step in range(args.syn_steps):
                        if not indices_chunks:
                            indices = torch.randperm(len(syn_images))
                            indices_chunks = list(torch.split(indices, args.batch_syn))
                        these_indices = indices_chunks.pop()
                        x = syn_images[these_indices]
                        this_y = y_hat[these_indices]
                        if args.texture:
                            x = torch.cat([torch.stack([torch.roll(im, (torch.randint(im_size[0]*args.canvas_size, (1,)), torch.randint(im_size[1]*args.canvas_size, (1,))), (1,2))[:,:im_size[0],:im_size[1]] for im in x]) for _ in range(args.canvas_samples)])
                            this_y = torch.cat([this_y for _ in range(args.canvas_samples)])
                        if args.dsa and (not args.no_aug):
                            x = DiffAugment(x, args.dsa_strategy, param=args.dsa_param)
                        if args.distributed and args.model[-2:]!='BN':
                            forward_params = student_params[-1].unsqueeze(0).expand(torch.cuda.device_count(), -1)
                        else:
                            forward_params = student_params[-1]
                        x = student_net(x, flat_param=forward_params)
                        ce_loss = nn.CrossEntropyLoss().to(torch.device("cuda:0"))(x, this_y)
                        # ce_loss = criterion(x, this_y)
                        grad = torch.autograd.grad(ce_loss, student_params[-1], create_graph=False)[0]
                        student_params.append(student_params[-1] - syn_lr * grad)
                    param_loss = torch.tensor(0.0).to(args.device)
                    param_dist = torch.tensor(0.0).to(args.device)
                    param_loss += torch.nn.functional.mse_loss(student_params[-1], target_params, reduction="sum")
                    param_dist += torch.nn.functional.mse_loss(starting_params, target_params, reduction="sum")
                    param_loss_list.append(param_loss)
                    param_dist_list.append(param_dist)
                    param_loss /= num_params
                    param_dist /= num_params
                    param_loss /= param_dist
                    grand_loss = param_loss
                    for _ in student_params:
                        del _
                    log_dict[str(i)] = grand_loss.detach().cpu()
                wandb.log(log_dict, step=it)


            for model_eval in model_eval_pool:
                print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))
                if args.dsa:
                    print('DSA augmentation strategy: \n', args.dsa_strategy)
                    print('DSA augmentation parameters: \n', args.dsa_param.__dict__)
                else:
                    print('DC augmentation parameters: \n', args.dc_aug_param)

                accs_test = []
                accs_train = []
                ''' Evaluate ema synthetic data '''
                ema_accs_test = []
                ema_accs_train = []

                for it_eval in range(args.num_eval):
                    net_eval = get_network(model_eval, channel, num_classes, im_size, dist=False).to(args.device) # get a random model
                    eval_labs = label_syn
                    with torch.no_grad():
                        image_save = image_syn
                    image_syn_eval, label_syn_eval = copy.deepcopy(image_save.detach()), copy.deepcopy(eval_labs.detach()) # avoid any unaware modification

                    args.lr_net = syn_lr.item()
                
                    # soft_label=generate_soft_label(expert_files, image_syn_eval, label_syn_eval)
                    # soft_cri = SoftCrossEntropy
                    # _, _, acc_test_soft_label = evaluate_synset(it_eval, net_eval, image_syn_eval, soft_label, testloader, args, texture=args.texture, train_criterion=soft_cri)
                    # wandb.log({'SoftAccuracy/{}'.format(model_eval): acc_test_soft_label}, step=it)

                    _, acc_train, acc_test = evaluate_synset(it_eval, copy.deepcopy(net_eval), image_syn_eval, label_syn_eval, testloader, args, texture=args.texture)
                    accs_test.append(acc_test)
                    accs_train.append(acc_train)


                    ''' Evaluate ema synthetic data '''
                    ema_eval_labs = label_syn
                    with torch.no_grad():
                        ema_image_syn = ema.module[0]
                        ema_image_save = ema_image_syn
                    ema_image_syn_eval, ema_label_syn_eval = copy.deepcopy(ema_image_save), copy.deepcopy(ema_eval_labs.detach()) # avoid any unaware modification
                    _, ema_acc_train, ema_acc_test = evaluate_synset(it_eval, copy.deepcopy(net_eval), ema_image_syn_eval, ema_label_syn_eval, testloader, args, texture=args.texture)
                    ema_accs_test.append(ema_acc_test)
                    ema_accs_train.append(ema_acc_train)


                    # soft_label=generate_soft_label(expert_files, ema_image_syn_eval, ema_label_syn_eval)
                    # soft_cri = SoftCrossEntropy
                    # _, _, ema_acc_test_soft_label = evaluate_synset(it_eval, net_eval, ema_image_syn_eval, soft_label, testloader, args, texture=args.texture, train_criterion=soft_cri)
                    # wandb.log({'ema_SoftAccuracy/{}'.format(model_eval): ema_acc_test_soft_label}, step=it)


                accs_test = np.array(accs_test)
                accs_train = np.array(accs_train)
                acc_test_mean = np.mean(accs_test)
                acc_test_std = np.std(accs_test)

                if acc_test_mean > best_acc[model_eval]:
                    best_acc[model_eval] = acc_test_mean
                    best_std[model_eval] = acc_test_std
                    save_this_it = True


                ''' Evaluate ema synthetic data '''
                ema_accs_test = np.array(ema_accs_test)
                ema_accs_train = np.array(ema_accs_train)
                ema_acc_test_mean = np.mean(ema_accs_test)
                ema_acc_test_std = np.std(ema_accs_test)

                if ema_acc_test_mean > ema_best_acc[model_eval]:
                    ema_best_acc[model_eval] = ema_acc_test_mean
                    ema_best_std[model_eval] = ema_acc_test_std
                    save_this_it = True

                

                print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs_test), model_eval, acc_test_mean, acc_test_std))
                wandb.log({'Accuracy/{}'.format(model_eval): acc_test_mean}, step=it)
                wandb.log({'Max_Accuracy/{}'.format(model_eval): best_acc[model_eval]}, step=it)
                wandb.log({'Std/{}'.format(model_eval): acc_test_std}, step=it)
                wandb.log({'Max_Std/{}'.format(model_eval): best_std[model_eval]}, step=it)


                ''' Evaluate ema synthetic data '''
                print('Evaluate ema %d random %s, ema_mean = %.4f ema_std = %.4f\n-------------------------'%(len(ema_accs_test), model_eval, ema_acc_test_mean, ema_acc_test_std))
                wandb.log({'ema_Accuracy/{}'.format(model_eval): ema_acc_test_mean}, step=it)
                wandb.log({'ema_Max_Accuracy/{}'.format(model_eval): ema_best_acc[model_eval]}, step=it)
                wandb.log({'ema_Std/{}'.format(model_eval): ema_acc_test_std}, step=it)
                wandb.log({'ema_Max_Std/{}'.format(model_eval): ema_best_std[model_eval]}, step=it)




        if it in eval_it_pool and (save_this_it or it % 1000 == 0):
            with torch.no_grad():
                image_save = image_syn.cuda()

                save_dir = os.path.join(".", "logged_files", args.dataset, str(args.ipc), 'FTD', args.model, wandb.run.name)

                if not os.path.exists(save_dir):
                    os.makedirs(os.path.join(save_dir,'Normal'))
                    os.makedirs(os.path.join(save_dir,'ema'))
                    

                torch.save(image_save.cpu(), os.path.join(save_dir, 'Normal',"images_{}.pt".format(it)))
                torch.save(label_syn.cpu(), os.path.join(save_dir, 'Normal', "labels_{}.pt".format(it)))
                torch.save(syn_lr.detach().cpu(), os.path.join(save_dir, 'Normal', "lr_{}.pt".format(it)))

                if save_this_it:
                    torch.save(image_save.cpu(), os.path.join(save_dir, 'Normal', "images_best.pt".format(it)))
                    torch.save(label_syn.cpu(), os.path.join(save_dir, 'Normal', "labels_best.pt".format(it)))
                    torch.save(syn_lr.detach().cpu(), os.path.join(save_dir, 'Normal', "lr_best.pt".format(it)))

                wandb.log({"Pixels": wandb.Histogram(torch.nan_to_num(image_syn.detach().cpu()))}, step=it)

                if args.ipc < 50 or args.force_save:
                    upsampled = image_save
                    if args.dataset != "ImageNet":
                        upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                        upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                    grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                    wandb.log({"Synthetic_Images": wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=it)
                    wandb.log({'Synthetic_Pixels': wandb.Histogram(torch.nan_to_num(image_save.detach().cpu()))}, step=it)

                    for clip_val in [2.5]:
                        std = torch.std(image_save)
                        mean = torch.mean(image_save)
                        upsampled = torch.clip(image_save, min=mean-clip_val*std, max=mean+clip_val*std)
                        if args.dataset != "ImageNet":
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                        grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                        wandb.log({"Clipped_Synthetic_Images/std_{}".format(clip_val): wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=it)

                    if args.zca:
                        image_save = image_save.to(args.device)
                        image_save = args.zca_trans.inverse_transform(image_save)
                        image_save.cpu()

                        torch.save(image_save.cpu(), os.path.join(save_dir, "images_zca_{}.pt".format(it)))

                        upsampled = image_save
                        if args.dataset != "ImageNet":
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                        grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                        wandb.log({"Reconstructed_Images": wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=it)
                        wandb.log({'Reconstructed_Pixels': wandb.Histogram(torch.nan_to_num(image_save.detach().cpu()))}, step=it)

                        for clip_val in [2.5]:
                            std = torch.std(image_save)
                            mean = torch.mean(image_save)
                            upsampled = torch.clip(image_save, min=mean - clip_val * std, max=mean + clip_val * std)
                            if args.dataset != "ImageNet":
                                upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                                upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                            grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                            wandb.log({"Clipped_Reconstructed_Images/std_{}".format(clip_val): wandb.Image(
                                torch.nan_to_num(grid.detach().cpu()))}, step=it)


        ''' Save ema synthetic data '''
        if it in eval_it_pool and (save_this_it or it % 1000 == 0):
            with torch.no_grad():
                ema_image_save = ema_image_syn.cuda()

                torch.save(ema_image_save.cpu(), os.path.join(save_dir, 'ema', "ema_images_{}.pt".format(it)))
                torch.save(label_syn.cpu(), os.path.join(save_dir, 'ema', "ema_labels_{}.pt".format(it)))
                torch.save(syn_lr.detach().cpu(), os.path.join(save_dir, 'ema', "ema_lr_{}.pt".format(it)))

                if save_this_it:
                    torch.save(ema_image_save.cpu(), os.path.join(save_dir, 'ema', "ema_images_best.pt".format(it)))
                    torch.save(label_syn.cpu(), os.path.join(save_dir, 'ema', "ema_labels_best.pt".format(it)))
                    torch.save(syn_lr.detach().cpu(), os.path.join(save_dir, 'ema', "ema_lr_best.pt".format(it)))

                wandb.log({"ema_Pixels": wandb.Histogram(torch.nan_to_num(ema_image_syn.detach().cpu()))}, step=it)

                if args.ipc < 50 or args.force_save:
                    upsampled = ema_image_save
                    if args.dataset != "ImageNet":
                        upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                        upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                    grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                    wandb.log({"ema_Synthetic_Images": wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=it)
                    wandb.log({'ema_Synthetic_Pixels': wandb.Histogram(torch.nan_to_num(ema_image_save.detach().cpu()))}, step=it)

                    for clip_val in [2.5]:
                        std = torch.std(ema_image_save)
                        mean = torch.mean(ema_image_save)
                        upsampled = torch.clip(ema_image_save, min=mean-clip_val*std, max=mean+clip_val*std)
                        if args.dataset != "ImageNet":
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                        grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                        wandb.log({"ema_Clipped_Synthetic_Images/std_{}".format(clip_val): wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=it)

                    if args.zca:
                        ema_image_save = ema_image_save.to(args.device)
                        ema_image_save = args.zca_trans.inverse_transform(ema_image_save)
                        ema_image_save.cpu()

                        torch.save(ema_image_save.cpu(), os.path.join(save_dir, "images_zca_{}.pt".format(it)))

                        upsampled = ema_image_save
                        if args.dataset != "ImageNet":
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                        grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                        wandb.log({"ema_Reconstructed_Images": wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=it)
                        wandb.log({'ema_Reconstructed_Pixels': wandb.Histogram(torch.nan_to_num(ema_image_save.detach().cpu()))}, step=it)

                        for clip_val in [2.5]:
                            std = torch.std(ema_image_save)
                            mean = torch.mean(ema_image_save)
                            upsampled = torch.clip(ema_image_save, min=mean - clip_val * std, max=mean + clip_val * std)
                            if args.dataset != "ImageNet":
                                upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                                upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                            grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                            wandb.log({"ema_Clipped_Reconstructed_Images/std_{}".format(clip_val): wandb.Image(
                                torch.nan_to_num(grid.detach().cpu()))}, step=it)






        wandb.log({"Synthetic_LR": syn_lr.detach().cpu()}, step=it)

        student_net = get_network(args.model, channel, num_classes, im_size, dist=False).to(args.device)  # get a random model

        student_net = ReparamModule(student_net)

        if args.distributed:
            student_net = torch.nn.DataParallel(student_net)

        student_net.train()

        num_params = sum([np.prod(p.size()) for p in (student_net.parameters())])

        if args.load_all:
            expert_trajectory = buffer[np.random.randint(0, len(buffer))]
        else:
            expert_trajectory = buffer[buffer_id[expert_idx]]

            # current = str(expert_id[file_idx])+','+str(buffer_id[expert_idx])+','

            expert_idx += 1
            if expert_idx == len(buffer):
                expert_idx = 0
                file_idx += 1
                if file_idx == len(expert_files):
                    file_idx = 0
                    random.shuffle(expert_id)
                print("loading file {}".format(expert_files[expert_id[file_idx]]))
                if args.max_files != 1:
                    del buffer
                    buffer = torch.load(expert_files[expert_id[file_idx]])
                if args.max_experts is not None:
                    buffer = buffer[:args.max_experts]
                random.shuffle(buffer_id)

        # Upper_Bound = 10 + int((args.max_start_epoch-10) * it/(args.Iteration))
        # Upper_Bound = min(Upper_Bound, args.max_start_epoch)
        # start_epoch = np.random.randint(0, Upper_Bound)
        start_epoch = np.random.randint(args.min_start_epoch, args.max_start_epoch)

        starting_params = expert_trajectory[start_epoch]

        target_params = expert_trajectory[start_epoch+args.expert_epochs]
        target_params = torch.cat([p.data.to(args.device).reshape(-1) for p in target_params], 0)

        student_params = [torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0).requires_grad_(True)]

        starting_params = torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0)

        syn_images = image_syn

        y_hat = label_syn.to(args.device)

        param_loss_list = []
        param_dist_list = []
        indices_chunks = []

        # current += str(start_epoch)
        # if current in expert_step_store:
        #     expert_step_store[current]+=1
        # else:
        #     expert_step_store[current]=1

        for step in range(args.syn_steps):

            if not indices_chunks:
                indices = torch.randperm(len(syn_images))
                indices_chunks = list(torch.split(indices, args.batch_syn))

            these_indices = indices_chunks.pop()


            x = syn_images[these_indices]
            this_y = y_hat[these_indices]

            if args.texture:
                x = torch.cat([torch.stack([torch.roll(im, (torch.randint(im_size[0]*args.canvas_size, (1,)), torch.randint(im_size[1]*args.canvas_size, (1,))), (1,2))[:,:im_size[0],:im_size[1]] for im in x]) for _ in range(args.canvas_samples)])
                this_y = torch.cat([this_y for _ in range(args.canvas_samples)])

            if args.dsa and (not args.no_aug):
                x = DiffAugment(x, args.dsa_strategy, param=args.dsa_param)

            if args.distributed:
                forward_params = student_params[-1].unsqueeze(0).expand(torch.cuda.device_count(), -1)
            else:
                forward_params = student_params[-1]
            x = student_net(x, flat_param=forward_params)
            ce_loss = criterion(x, this_y)

            grad = torch.autograd.grad(ce_loss, student_params[-1], create_graph=True)[0]

            student_params.append(student_params[-1] - syn_lr * grad)

        param_loss = torch.tensor(0.0).to(args.device)
        param_dist = torch.tensor(0.0).to(args.device)

        param_loss += torch.nn.functional.mse_loss(student_params[-1], target_params, reduction="sum")
        param_dist += torch.nn.functional.mse_loss(starting_params, target_params, reduction="sum")

        param_loss_list.append(param_loss)
        param_dist_list.append(param_dist)


        param_loss /= num_params
        param_dist /= num_params

        param_loss /= param_dist

        grand_loss = param_loss

        optimizer_img.zero_grad()
        optimizer_lr.zero_grad()

        grand_loss.backward()

        optimizer_img.step()
        optimizer_lr.step()


        ###using EMA
        # Update the moving average with the new parameters from the last optimizer step
        ema.update([image_syn])


        wandb.log({"Grand_Loss": grand_loss.detach().cpu(),
                   "Start_Epoch": start_epoch})

        # wandb.log({"sample" : wandb.plot.line_series(
        # xs=expert_step_store[current],
        # ys=grand_loss,
        # keys=current,
        # title="Loss Statistic")})


        for _ in student_params:
            del _

        if it%10 == 0:
            print('%s iter = %04d, loss = %.4f' % (get_time(), it, grand_loss.item()))

    wandb.finish()


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

    parser.add_argument('--lr_img', type=float, default=1000, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_lr', type=float, default=1e-05, help='learning rate for updating... learning rate')
    parser.add_argument('--lr_teacher', type=float, default=0.01, help='initialization for synthetic learning rate')

    parser.add_argument('--lr_init', type=float, default=0.01, help='how to init lr (alpha)')

    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_syn', type=int, default=None, help='should only use this if you run out of VRAM')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')

    parser.add_argument('--pix_init', type=str, default='real', choices=["noise", "real"],
                        help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')

    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                        help='whether to use differentiable Siamese augmentation.')

    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')

    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--buffer_path', type=str, default='./buffers', help='buffer path')

    parser.add_argument('--expert_epochs', type=int, default=3, help='how many expert epochs the target params are')
    parser.add_argument('--syn_steps', type=int, default=20, help='how many steps to take on synthetic data')
    parser.add_argument('--max_start_epoch', type=int, default=25, help='max epoch we can start at')
    parser.add_argument('--min_start_epoch', type=int, default=0, help='max epoch we can start at')

    parser.add_argument('--zca', action='store_true', help="do ZCA whitening")

    parser.add_argument('--load_all', action='store_true', help="only use if you can fit all expert trajectories into RAM")

    parser.add_argument('--no_aug', type=bool, default=False, help='this turns off diff aug during distillation')

    parser.add_argument('--texture', action='store_true', help="will distill textures instead")
    parser.add_argument('--canvas_size', type=int, default=2, help='size of synthetic canvas')
    parser.add_argument('--canvas_samples', type=int, default=1, help='number of canvas samples per iteration')


    parser.add_argument('--max_files', type=int, default=None, help='number of expert files to read (leave as None unless doing ablations)')
    parser.add_argument('--max_experts', type=int, default=None, help='number of experts to read per file (leave as None unless doing ablations)')

    parser.add_argument('--force_save', action='store_true', help='this will save images for 50ipc')
    parser.add_argument('--ema_decay', type=float, default=0.999)
    parser.add_argument('--project', type=str, default='TEST', help='WanDB Project Name')

    parser.add_argument('--record_loss', type=bool, default=False, help='If record loss')
    parser.add_argument('--skip_first_eva', type=bool, default=False, help='If skip first eva')
    parser.add_argument('--parall_eva', type=bool, default=False, help='If skip first eva')
    parser.add_argument('--res', type=int, default=32, help='If skip first eva')

    args = parser.parse_args()

    main(args)


