# -*- coding: utf-8 -*-
import os
import gc
import copy
import torch
import pickle
import random
import argparse
import torchvision
import numpy as np
import pandas as pd
import torch.nn as nn
from PIL import Image
import torch.optim as optim
from netcal.metrics import ECE
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torchvision import transforms
from torch.utils.data import Dataset
from models.densenet import DenseNet
from models.inception import inceptionv4
from models.resnet import ResNet18, ResNet50
from utils.utils import progress_bar, save_model
from torch.utils.data.sampler import SubsetRandomSampler
###################################################################################
#################################### Arguments ####################################
###################################################################################
np.random.seed(42)
parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--ce', default = True, help='Cross entropy use')
parser.add_argument('--dataset', default = 'fashionmnist', help='Dataset')
parser.add_argument('--num_classes', default = 10, help='Dataset')
parser.add_argument('--batch_size', default = 512, help='Batch size')
parser.add_argument('--epochs', default = 100, help='Number of epochs')
parser.add_argument('--estop', default = 10, help='Early stopping')
parser.add_argument('--ece_bins', default = 10, help='ECE bins')
parser.add_argument('--tr_size', default = 0.8, help='Training/Validation sets split ratio')
parser.add_argument('--model', default = 'densenet', help='Model to be used')
args = parser.parse_args([])
device = 'cuda' if torch.cuda.is_available() else 'cpu'

######################################################################################
#################################### Data_Loaders ####################################
######################################################################################
def get_transforms(dataset):
    if dataset[:5] == 'cifar':
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), 
                                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
        transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
        transform_mi = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), 
                                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
    else:
        transform_train = transforms.Compose([transforms.Resize(96), transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,)),])
        transform_test = transforms.Compose([transforms.Resize(96),transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,)),])    
        transform_mi = transforms.Compose([transforms.Resize(128), transforms.RandomCrop(88, padding=4), transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),])
    return transform_train, transform_test, transform_mi

transform_train, transform_test, transform_mi = get_transforms(args.dataset)
if args.dataset == 'fashionmnist':
    trainset = torchvision.datasets.FashionMNIST(root='data', train=True, download=True, transform=transform_train,)
    validset = torchvision.datasets.FashionMNIST(root='data', train=True, download=True, transform=transform_test,)
    testset  = torchvision.datasets.FashionMNIST(root='data', train=False, download=True, transform=transform_test,)
    trainset2  = torchvision.datasets.FashionMNIST(root='data', train=True , download=True)
    testset2   = torchvision.datasets.FashionMNIST(root='data', train=False, download=True)
elif args.dataset == 'cifar10':
    trainset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform_train,)
    validset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform_test,)
    testset  = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=transform_test,)
    trainset2  = torchvision.datasets.CIFAR10(root='data', train=True , download=True)
    testset2   = torchvision.datasets.CIFAR10(root='data', train=False, download=True)
elif args.dataset == 'cifar100':
    trainset = torchvision.datasets.CIFAR100(root='data', train=True, download=True, transform=transform_train,)
    validset = torchvision.datasets.CIFAR100(root='data', train=True, download=True, transform=transform_test,)
    testset  = torchvision.datasets.CIFAR100(root='data', train=False, download=True, transform=transform_test,)
    trainset2  = torchvision.datasets.CIFAR100(root='data', train=True , download=True)
    testset2   = torchvision.datasets.CIFAR100(root='data', train=False, download=True)

x_train = trainset2.data; x_test = testset2.data
y_train = trainset2.targets; y_test = testset2.targets
indices = list(range(len(trainset)))
split = int(np.floor(args.tr_size * len(trainset)))
np.random.shuffle(indices)
train_idx, valid_idx = indices[:split], indices[split:]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)
trainloader = torch.utils.data.DataLoader(trainset, sampler=train_sampler, batch_size=args.batch_size, shuffle=False)
validloader = torch.utils.data.DataLoader(validset, sampler=valid_sampler, batch_size=args.batch_size, shuffle=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)
trials = 1
#######################################################################################
#################################### Temp Scaling #####################################
#######################################################################################
class ModelWithTemperature(nn.Module):
    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    def set_temperature(self, validloader):
        self.cuda()
        nll_criterion = nn.CrossEntropyLoss().cuda()
        logits_list = []; labels_list = []
        with torch.no_grad():
            for input, label in validloader:
                input = input.cuda()
                logits = self.model(input)
                logits_list.append(logits); labels_list.append(label)
            logits = torch.cat(logits_list).cuda(); labels = torch.cat(labels_list).cuda()
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        def eval():
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss
        optimizer.step(eval)
        return self

########################################################################################
#################################### Model Trainer #####################################
########################################################################################
class ModelTrainer():
    def __init__(self, net, trainloader, validloader, optimizer, scheduler, criterion_train, criterion_test, save_path, 
                 epochs = args.epochs, estop = args.estop, device = 'cuda', num_classes = args.num_classes, trials = trials):
        self.net = net
        self.best_net = net
        self.trainloader = trainloader
        self.validloader = testloader
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion_train
        self.criterion1 = criterion_test
        self.save_path = save_path
        self.num_classes = num_classes
        self.best_loss = 1e9
        self.early = estop
        self.trials = trials
        for epoch in range(epochs):
            self.train(epoch)
            self.valid(epoch)
            if self.early > estop:
                break

    def train(self, epoch):
        print('\nEpoch: %d' % epoch)
        self.net.train()
        train_loss = 0; correct = 0; total = 0
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            try:
                loss = self.criterion(outputs, targets)
            except:
                loss = self.criterion(outputs, targets, inputs)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar(batch_idx, len(self.trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % 
                (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        self.scheduler.step()
        print('Train Loss:', train_loss)

    def valid(self, epoch):
        self.net.eval()
        self.valid_loss = 0; total = 0; correct = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.validloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.net(inputs)
                loss = self.criterion1(outputs, targets)
                self.valid_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                progress_bar(batch_idx, len(self.validloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % 
                    (self.valid_loss/(batch_idx+1), 100.*correct/total, correct, total))
        acc = 100.*correct/total
        print('Validation Accuracy: ', acc)
        if self.valid_loss < self.best_loss:
            print('Saving..')
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            self.best_net = copy.deepcopy(self.net) 
            self.best_loss = self.valid_loss
            self.early = 0
        else:
            self.early += 1

#########################################################################################
#################################### Model selector #####################################
#########################################################################################
def get_new_model(tmp_scale = True, num_classes = args.num_classes):
    if args.model == 'resnet18':
        return ResNet18(tmp_scale = tmp_scale, num_classes = num_classes)
    elif args.model == 'resnet50':
        return ResNet50(tmp_scale = tmp_scale, num_classes = num_classes)
    elif args.model == 'resnet101':
        return ResNet101(tmp_scale = tmp_scale, num_classes = num_classes)
    elif args.model == 'inceptionv4':
        return inceptionv4(tmp_scale = tmp_scale, num_classes = num_classes)
    elif args.model == 'densenet':
        return DenseNet(tmp_scale = tmp_scale)

criterion1 = nn.CrossEntropyLoss()
results_dict = {}
results_df = pd.DataFrame()

for trial in range(3):
    gc.collect()
    main_path = 'files_' + args.model + '_' + args.dataset + '/trial' + str(trial) + '/' 
    if not os.path.exists(main_path):
        os.makedirs(main_path)
    ##########################################################################################
    #################################### Train CE Network ####################################
    ##########################################################################################
    title = "CrossEntropy"; criterion = criterion1; save_path = main_path + 'CrossEntropy.bin'
    if os.path.isfile(save_path):
        print ("Model exists....Continue")
    else:
        print ("Model does not exist!....Training now:", title)
        net = get_new_model()
        net = net.to(device)
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001, nesterov= True)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,60,90])
        model = ModelTrainer(net = net, trainloader = trainloader, validloader = validloader, optimizer = optimizer, scheduler = scheduler, 
                    criterion_train = criterion, criterion_test = criterion1, save_path = save_path, device = device)
        if args.mutual_info:
            model.mutual_info_values = np.add(model.mutual_info_values, -min(model.mutual_info_values))
            with open(main_path + title+'.pkl', 'wb') as fp:
                pickle.dump(model.mutual_info_values, fp)
        
        save_model(model.best_net, save_path)
        scaled_model = ModelWithTemperature(model.best_net)
        scaled_model.set_temperature(validloader)
        save_path = main_path + 'CETmpScaling.bin'
        model.best_net.temperature = scaled_model.temperature
        save_model(model.best_net, save_path)