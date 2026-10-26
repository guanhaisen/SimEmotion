# -*- coding: UTF-8 -*-
# train.py
#!/usr/bin/env	python3

""" train network using pytorch

author baiyu
"""
import copy
import os
import sys
import argparse
from polar_coordinates import Polar_coordinates
import time
from datetime import datetime
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torchvision import models
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter

from conf import settings

import torch.nn.functional as F
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights,SemihardNegativeTripletSelector

import torch
import copy

def list_mle(y_true, y_pred):
    # shuffle for randomised tie resolution
    random_indices = torch.randperm(y_pred.shape[-1])
    y_pred_shuffled = y_pred[:, random_indices]
    y_true_shuffled = y_true[:, random_indices]

    y_true_sorted, indices = y_true_shuffled.sort(descending=True, dim=-1)
    mask = y_true_sorted == -1
    preds_sorted_by_true = torch.gather(y_pred_shuffled, dim=1, index=indices)
    preds_sorted_by_true[mask] = float('-inf')
    max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)
    preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values
    cumsums = torch.cumsum(preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1).flip(dims=[1])
    observation_loss = torch.log(cumsums + 1e-7) - preds_sorted_by_true_minus_max
    observation_loss[mask] = 0.0

    return torch.mean(torch.sum(observation_loss, dim=1))




def train(epoch,args,loss_fn1):

    start = time.time()
    net.train()
    net.cuda(args.gpuid)
    for batch_index, (images, labels) in enumerate(sll_train_loader):

        if args.gpu:
            labels = labels.cuda(args.gpuid)
            images = images.cuda(args.gpuid)
            

        optimizer.zero_grad()
        outputs = net(images)
        if args.dataset_type=="ldl":
            _, singlelabel = torch.max(labels, dim=1)
            loss =loss_function(outputs,singlelabel)
            # outputs=torch.softmax(outputs,dim = 1)
            # loss = loss_function(torch.log(outputs), labels)
        elif args.dataset_type=="sll":
            loss = loss_function(outputs, labels)
            
        if args.loss=="list_mle":
            if args.dataset=='FI' or args.dataset=='EmotionSet':
                emotion_entropy=[
                [8,7,6,5,1,2,3,4],
                [7,8,7,6,2,1,2,3],
                [6,7,8,7,3,2,1,2],
                [5,6,7,8,4,3,2,1],
                [1,2,3,4,8,7,6,5],
                [2,1,2,3,7,8,7,6],
                [3,2,1,2,6,7,8,7],
                [4,3,2,1,5,6,7,8]]
                for i in range(8):
                    for j in range(8):
                        emotion_entropy[i][j]=emotion_entropy[i][j]-1

                emotion_entropy=np.array(emotion_entropy)
                emotion_entropy= torch.tensor(emotion_entropy,dtype=torch.float32)
                emotion_entropy=emotion_entropy.cuda(args.gpuid)
                rankloss=list_mle(emotion_entropy[labels],outputs)
                
                loss+=0.2*rankloss
                
        
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(sll_train_loader) + batch_index + 1

        last_layer = list(net.children())[-1]
        for name, para in last_layer.named_parameters():
            if 'weight' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
            if 'bias' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)
        if batch_index%10==0:
            print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
                loss.item(),
                optimizer.param_groups[0]['lr'],
                epoch=epoch,
                trained_samples=batch_index * args.b + len(images),
                total_samples=len(sll_train_loader.dataset)
            ))
        # with open('result.txt', 'a') as f:
        #     f.write(str(loss.item())+'\n')
    

        #update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)

        if epoch <= args.warm:
            warmup_scheduler.step()

    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))



@torch.no_grad()
def eval_training(dataset,epoch=0, tb=True):
    if dataset=='FI' or args.dataset=='EmotionSet':
        emotion_entropy=[[1,2,3,4,8,7,6,5],
            [2,1,2,3,7,8,7,6],
            [3,2,1,2,6,7,8,7],
            [4,3,2,1,5,6,7,8],
            [8,7,6,5,1,2,3,4],
            [7,8,7,6,2,1,2,3],
            [6,7,8,7,3,2,1,2],
            [5,6,7,8,4,3,2,1]]
        emotion_entropy_noacc=copy.deepcopy(emotion_entropy)
        for i in range(8):
            for j in range(8):
                if i!=j:
                    emotion_entropy_noacc[i][j]-=1

        
        
    emotion_entropy=np.array(emotion_entropy)
    # distance=1/(emotion_entropy*emotion_entropy)
    distance=1/(emotion_entropy)
    distance= torch.tensor(distance,dtype=torch.float32)
    distance=distance.cuda(args.gpuid) 
    
    emotion_entropy_noacc=np.array(emotion_entropy_noacc)
    distance_noacc=1/(emotion_entropy_noacc)
    distance_noacc= torch.tensor(distance_noacc,dtype=torch.float32)
    distance_noacc=distance_noacc.cuda(args.gpuid) 
    
    confusion_matrix=torch.zeros_like(distance)
    emotion_entropy_score=0
    start = time.time()
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    topk_acc=[0,0,0,0,0]
    for (images, labels) in sll_test_loader:
    
        if args.gpu:
            images = images.cuda(args.gpuid)
            labels = labels.cuda(args.gpuid)
       
        outputs = net(images)
        
        
        loss = loss_function(outputs, labels)
        
        test_loss += loss.item()
        _, preds = outputs.max(1)
        
        prob = F.softmax(outputs,dim=0)

        topkacc_b=accuracy(outputs,labels)
        
        topk_acc = [i + j for i, j in zip(topk_acc, topkacc_b)]
       
        emotion_entropy_score+=torch.sum(distance[labels]*prob,dim=1).sum()
        correct += preds.eq(labels).sum()
        for i in range(len(labels)):
            confusion_matrix[labels[i]][preds[i]]+=1
    topk_acc=[i/len(sll_test_loader.dataset) for i in topk_acc]
     
    
    finish = time.time()

    acc_EE=(confusion_matrix*distance).sum()
    noacc_confusion_matrix=confusion_matrix.clone()
    for i in range(len(noacc_confusion_matrix)):
        noacc_confusion_matrix[i,i]=0
    
    noacc_EE=(noacc_confusion_matrix*distance_noacc).sum()
    print(distance)
    print(distance_noacc)
    print('Evaluating Network.....')
    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f},Emotion_Entropy: {:.4f},acc_EE: {:.4f},noacc_EE: {:.4f},Time consumed:{:.2f}s'.format(
        epoch,
        test_loss / len(sll_test_loader.dataset),
        correct.float() / len(sll_test_loader.dataset),
        emotion_entropy_score/len(sll_test_loader.dataset),
        acc_EE / len(sll_test_loader.dataset),
        noacc_EE / (len(sll_test_loader.dataset)-correct.float()),
        finish - start
    ))
    print()
    
    #add informations to tensorboard
    if tb:
        writer.add_scalar('Test/Average loss', test_loss / len(sll_test_loader.dataset), epoch)
        writer.add_scalar('Test/Accuracy', correct.float() / len(sll_test_loader.dataset), epoch)

    return correct.float() / len(sll_test_loader.dataset),emotion_entropy_score.float() / len(sll_test_loader.dataset),acc_EE / len(sll_test_loader.dataset),noacc_EE / (len(sll_test_loader.dataset)-correct.float()),topk_acc


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default='resnet50',help='net type')
    parser.add_argument('-gpu',type=bool,default=True, help='use gpu or not')
    parser.add_argument('-b', type=int, default=32, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-numclass', type=int, default=8, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')

    parser.add_argument('--path', type=str, required=True, help='dataset path')
    parser.add_argument('--dataset', type=str, default='FI',help='FI,EmoSet')
    parser.add_argument('--gpuid', type=int, default=0,help='net type')
    parser.add_argument('--loss',type=str,default='original',help='topk,topk_neighbor,polar,original,circle_end,list_mle,mixloss')
    parser.add_argument('--network',type=str,default='resnet50',help='resnet18,resnet50,resnet101')
    parser.add_argument('-dataset_type',type=str,default='sll')
    parser.add_argument('-alpha',type=float,default=0.2)
    parser.add_argument('-beta',type=float,default=1.0)
    parser.add_argument('--seed',type=int,default=0)
           

    if args.network=='resnet50':
        if 'UnbiasedEmo' in args.dataset:
            args.lr=0.01
        net=models.resnet50(pretrained=True)
        net.fc = nn.Linear(2048, args.numclass)
    elif args.network=='resnet101':
        if 'UnbiasedEmo' in args.dataset:
            args.lr=0.01
        net=models.resnet101(pretrained=True)
        net.fc = nn.Linear(2048, args.numclass)
    elif args.network=='resnet18':
        net=models.resnet18(pretrained=True)
        net.fc = nn.Linear(512, args.numclass)
    net.cuda(args.gpuid)
    
    set_seed(0)
    
    #data preprocessing:
    sll_train_loader = get_training_dataloader(
        args,
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )

    sll_test_loader = get_test_dataloader(
        args,
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )


    #criterion_JS=nn.functional.kl_div()
    if args.dataset_type=='ldl':
        
        #loss_function = torch.nn.KLDivLoss(reduction ='batchmean')
        loss_function = nn.CrossEntropyLoss()
    else:     
        loss_function = nn.CrossEntropyLoss()
    
    
    # criterion = nn.KLDivLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
    #optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(sll_train_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
    
    if args.resume:
        recent_folder = most_recent_folder(os.path.join(settings.CHECKPOINT_PATH, args.network), fmt=settings.DATE_FORMAT)
        if not recent_folder:
            raise Exception('no recent folder were found')

        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.network, recent_folder)

    else:
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH,args.dataset,args.network,args.loss,str(args.alpha)+'_'+str(args.beta))

    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)

    #since tensorboard can't overwrite old values
    #so the only way is to create a new tensorboard log
    writer = SummaryWriter(log_dir=os.path.join(
            settings.LOG_DIR, args.network, settings.TIME_NOW))
    input_tensor = torch.Tensor(1, 3, 32, 32)
    if args.gpu:
        input_tensor = input_tensor.cuda(args.gpuid)
    writer.add_graph(net, input_tensor)

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{type}.pth')
    #net = nn.DataParallel(net.cuda(), device_ids=[0,1], output_device=0)
    best_acc = 0.0
    if args.resume:
        best_weights = best_acc_weights(os.path.join(settings.CHECKPOINT_PATH, args.network, recent_folder))
        if best_weights:
            weights_path = os.path.join(settings.CHECKPOINT_PATH, args.network, recent_folder, best_weights)
            print('found best acc weights file:{}'.format(weights_path))
            print('load best training file to test acc...')
            net.load_state_dict(torch.load(weights_path))
            best_acc,emo_entropy,acc_EE,noacc_EE = eval_training(tb=False)
            print('best acc is {:0.2f}'.format(best_acc))

        recent_weights_file = most_recent_weights(os.path.join(settings.CHECKPOINT_PATH, args.network, recent_folder))
        if not recent_weights_file:
            raise Exception('no recent weights file were found')
        weights_path = os.path.join(settings.CHECKPOINT_PATH, args.network, recent_folder, recent_weights_file)
        print('loading weights file {} to resume training.....'.format(weights_path))
        net.load_state_dict(torch.load(weights_path))

        resume_epoch = last_epoch(os.path.join(settings.CHECKPOINT_PATH, args.network, recent_folder))

    best_acc=0
    best_acc_EE=0
    best_noacc_EE=0
    for epoch in range(1, settings.EPOCH + 1):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        if args.resume:
            if epoch <= resume_epoch:
                continue

        train(epoch,args,loss_fn1)
        if args.dataset_type=="ldl":
            acc,emo_entropy,acc_EE,noacc_EE,LDL_metric,ldl_score,ldl_score2 = ldl_eval_training(args.dataset,epoch)
        else:    
            acc,emo_entropy,acc_EE,noacc_EE,topkacc = eval_training(args.dataset,epoch)
        
        #mll_metric=validate_mll_zcx(sll_test_loader,net,args)
        savemodelname=str(args.dataset)+'_'+str(args.network)+'_0.2'+str(args.loss)
        if best_acc<acc:
            best_acc=acc
            weights_path = checkpoint_path.format(net=args.network,type='bestacc_'+savemodelname+str(args.noise_type)+'_'+str(args.noise_ratio)+"_"+str(args.repeat))
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)
        if best_acc_EE<acc_EE:
            best_acc_EE=acc_EE
            weights_path = checkpoint_path.format(net=args.network, type='bestaccEE_'+savemodelname+str(args.noise_type)+'_'+str(args.noise_ratio)+"_"+str(args.repeat))
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)
        if best_noacc_EE<noacc_EE:
            best_noacc_EE=noacc_EE
            weights_path = checkpoint_path.format(net=args.network, type='bestnoaccEE_'+savemodelname+str(args.noise_type)+'_'+str(args.noise_ratio)+"_"+str(args.repeat))
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)
        
             
        file="experiment/"+str(args.dataset)+'/'+str(args.dataset)+'_'+str(args.lr)+'_'+str(args.b)+'_0.2'+str(args.loss)+'_'+str(args.network)+'_'+str(args.alpha)+'_'+str(args.beta)+'_'+str(args.noise_type)+'_'+str(args.noise_ratio)+"_"+str(args.repeat)+"_seed:"+str(args.seed)+'_SGD_sll.txt'
        
        if not os.path.exists("experiment/"+str(args.dataset)):
            os.makedirs("experiment/"+str(args.dataset)) 
        
        
        with open(file, 'a') as f:
            f.write(str(epoch)+"_acc:"+str(float(acc))+" "+' '+"acc_EE:"+str(float(acc_EE))+' '+"noacc_EE:"+str(float(noacc_EE))+' '+"best_acc:"+str(float(best_acc))+'\n')
            f.write('best_acc: {}, best_ECC: {}, best_EMC: {} \n'.format(float(best_acc),float(best_acc_EE),float(best_noacc_EE)))
       

    writer.close()







       
       
