# import os
import random

import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.modules.batchnorm import BatchNorm2d

import argparse

import matplotlib.pyplot as plt
import numpy as np
import math
import time

from any_quant import qfn, activation_quantize_fn, weight_quantize_fn, psum_quantize_fn, Activate, BatchNorm2d_Q, Linear_Q
from LSQ import LsqWeight, LsqPsum
from utils import split4d, im2col_weight, im2col_acti, weightTile, weightTile_new, weightTile_HxW
from resnet_quan import *

def main():

    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--a_bit', default=4, type=int, help='activation bit-width')
    parser.add_argument('--w_bit', default=4, type=int, help='weight bit-width')
    parser.add_argument('--split_bit', default=4, type=int, help='split bit-width')
    parser.add_argument('--ps_bit', default=4, type=int, help='partial-sum bit-width')
    parser.add_argument('--num_sigma', default=6, type=int, help='number of standard deviation')

    parser.add_argument('--w_mode', default='Layer', type=str, help='weight mode')
    parser.add_argument('--ps_mode', default='Array', type=str, help='partial-sum mode')

    parser.add_argument('--isRow', action='store_true', default=False, help='whether to row-direction tiling')
    parser.add_argument('--w_ch', action='store_true', default=False, help='whether to weight channel-wise')
    parser.add_argument('--ps_ch', action='store_true', default=False, help='whether to partial-sum channel-wise')
    parser.add_argument('--psumOpt', action='store_true', default=False, help='whether to quantize partial-sum')

    parser.add_argument('--arr_size', type=int, default=512, help='CIM array size')

    # Optimizer and scheduler
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--num_classes', default=10, type=int, help='number of classes')

    parser.add_argument('--batch', '--batch-size', "-b", default=128, type=int, metavar='N', help='batch size (default: 128)')
    parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')

    parser.add_argument('--optimizer', default="SGD", help='optimizer to use, e.g. SGD, Adam')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate for updating network parameters')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('-wd', '--weight_decay', default=5e-4, type=float, metavar='W', help='weight decay (default: 5e-4)', dest='weight_decay')
    parser.add_argument("--nesterov", default=True, type=str_to_bool, help="if set nesterov")
    parser.add_argument("--scheduler", default="CosineAnnealingLR", type=str, help="Learning rate scheduler")
    parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')


    args = parser.parse_args()


    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.dataset == 'CIFAR10':
        transform_train = transforms.Compose([
           transforms.RandomCrop(32, padding=4),  #resises the image so it can be perfect for our model.
           transforms.RandomHorizontalFlip(), # FLips the image w.r.t horizontal axis
           transforms.ToTensor(), # comvert the image to tensor so that it can work with torch
           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))])
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        trainset = torchvision.datasets.CIFAR10(root='./data', 
                                                train=True, 
                                                download=False, 
                                                transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', 
                                               train=False, 
                                               download=False, 
                                               transform=transform)

    elif args.dataset == 'CIFAR100':
       transform_train = transforms.Compose([
          transforms.RandomCrop(32, padding=4),  #resises the image so it can be perfect for our model.
          transforms.RandomHorizontalFlip(), # FLips the image w.r.t horizontal axis
          transforms.ToTensor(), # comvert the image to tensor so that it can work with torch
          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))])
       
       transform = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
       
       trainset = torchvision.datasets.CIFAR100(root='./data',
                                            train=True,
                                            download=True,
                                            transform=transform_train)
       testset = torchvision.datasets.CIFAR100(root='./data',
                                          train=False,
                                          download=True,
                                          transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch, shuffle=True, num_workers=args.workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch, shuffle=False, num_workers=args.workers)

    model = ResNet20_arr(a_bit=args.a_bit, w_bit=args.w_bit, split_bit=args.split_bit, w_mode=args.w_mode, ps_bit=args.ps_bit, num_sigma=args.num_sigma, psum_mode=args.ps_mode, block=BasicBlock_arr,
                     arr_size=args.arr_size, num_units=[3, 3, 3], num_classes=args.num_classes, isRow=args.isRow, w_per_ch=args.w_ch, ps_per_ch=args.ps_ch, psumOpt=args.psumOpt, expand=1).cuda()

    model.train()

    train_model(model=model, train_loader=trainloader, test_loader=testloader,
                device=args.device, learning_rate=args.lr, epochs=args.epochs)

def evaluate_model(model, test_loader, device, criterion=None):

    model.eval()
    model.to(device)

    running_loss = 0
    running_corrects = 0

    for inputs, labels in test_loader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        if criterion is not None:
            loss = criterion(outputs, labels).item()
        else:
            loss = 0

        # statistics
        running_loss += loss * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    eval_loss = running_loss / len(test_loader.dataset)
    eval_accuracy = running_corrects / len(test_loader.dataset)

    return eval_loss, eval_accuracy


def train_model(model, train_loader, test_loader, device, learning_rate=5e-3,
                 epochs=200):
    criterion = nn.CrossEntropyLoss()

    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=3e-3)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.1, last_epoch=-1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=70, eta_min=0, last_epoch= -1, verbose=False) # T_max ...

    best_eval = -1
    best_epoch = 0

    model.eval()
    eval_loss, eval_accuracy = evaluate_model(model=model, test_loader=test_loader,
                                            device=device, criterion=criterion)
    print("Epoch: {:03d} Eval Loss: {:.4f} Eval Acc: {:.4f}".format(0, eval_loss, eval_accuracy))

    trainLossTensor = torch.empty((epochs,), dtype=torch.float64)
    evalLossTensor = torch.empty((epochs,), dtype=torch.float64)
    epochTensor = torch.empty((epochs,), dtype=torch.int64)

    for epoch in range(epochs):
      # Training
      epochTensor[epoch] = epoch
      model.train()
      running_loss = 0
      running_corrects = 0

      for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        loss.backward()

        # # gradient clipping
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
      train_loss = running_loss / len(train_loader.dataset)
      train_accuracy = running_corrects / len(train_loader.dataset)
      trainLossTensor[epoch] = train_loss

      # Evaluation
      model.eval()
      eval_loss, eval_accuracy = evaluate_model(model=model,
                                              test_loader=test_loader,
                                              device=device, criterion=criterion)
      evalLossTensor[epoch] = eval_loss

      scheduler.step()
      print("Epoch: {:03d} Train Loss: {:.4f} Train Acc: {:.4f} Eval Loss: {:.4f} Eval Acc: {:.4f}".format(epoch+1, train_loss, train_accuracy, eval_loss, eval_accuracy))

      if eval_accuracy > best_eval:
        best_eval = eval_accuracy
        best_epoch = epoch+1

      if epoch%60 == 0:
        plt.figure(figsize=(5,3))
        plt.grid(color='k', linestyle='--', linewidth=1)
        plt.scatter(epochTensor[:epoch+1], trainLossTensor[:epoch+1], s=20, label='Train Loss')
        plt.scatter(epochTensor[:epoch+1], evalLossTensor[:epoch+1], s=20, label='Eval Loss')

        plt.title('Train Loss vs. Eval Loss')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend(loc='upper right')

        plt.show()

    print("Best eval accuracy: {:.4f} @ epoch {:03d}".format(best_eval, best_epoch))

    # plot
    plt.figure(figsize=(10,6))
    plt.grid(color='k', linestyle='--', linewidth=1)
    plt.scatter(epochTensor, trainLossTensor, s=20, c='m', label='Train Loss')
    plt.scatter(epochTensor, evalLossTensor, s=20, c='c', label='Eval Loss')

    plt.title('Train Loss vs. Eval Loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(loc='upper right')

    plt.show()

    return model


if __name__ == '__main__':
    main()
