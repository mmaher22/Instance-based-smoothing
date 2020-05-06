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
from densenet import DenseNet
from netcal.metrics import ECE
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from inception import inceptionv4
from torchvision import transforms
from torch.utils.data import Dataset
from utils import progress_bar, save_model
from resnet import ResNet18, ResNet50, ResNet101
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
parser.add_argument('--monte_carlo', default = 10, help='monte_carlo samples')
parser.add_argument('--tr_size', default = 0.8, help='Training/Validation sets split ratio')
parser.add_argument('--model', default = 'densenet', help='Model to be used')
parser.add_argument('--tsne', default = False, help='TSNE Plots')
parser.add_argument('--reliability', default = True, help='Reliability Plots')
parser.add_argument('--mutual_info', default = False, help='Mutual Information')
args = parser.parse_args([])
device = 'cuda' if torch.cuda.is_available() else 'cpu'
######################################################################################
#################################### Data_Loaders ####################################
######################################################################################
def get_class_i(x, y, i, maxi = 0):
        y = np.array(y)
        pos_i = np.argwhere(y == i)
        if maxi == 0:
            pos_i = list(pos_i[:,0])
        else:
            pos_i = list(pos_i[:maxi, 0])
        x_i = [x[j] for j in pos_i]
        return x_i

class DatasetMaker(Dataset):
    def __init__(self, datasets, transformFunc):
        """
        datasets: a list of get_class_i outputs, i.e. a list of list of images for selected classes
        """
        self.datasets = datasets
        self.lengths  = [len(d) for d in self.datasets]
        self.transformFunc = transformFunc
    def __getitem__(self, i):
        class_label, index_wrt_class = self.index_of_which_bin(self.lengths, i)
        img = self.datasets[class_label][index_wrt_class]
        img = Image.fromarray(img)
        seed = np.random.randint(2147483647) # make a seed with numpy generator 
        random.seed(seed) # apply this seed to img transforms
        img = self.transformFunc(img)
        return img, class_label
    def __len__(self):
        return sum(self.lengths)
    
    def index_of_which_bin(self, bin_sizes, absolute_index, verbose=False):
        """
        Given the absolute index, returns which bin it falls in and which element of that bin it corresponds to.
        """
        accum = np.add.accumulate(bin_sizes)
        if verbose:
            print("accum =", accum)
        bin_index  = len(np.argwhere(accum <= absolute_index))
        if verbose:
            print("class_label =", bin_index)
        index_wrt_class = absolute_index - np.insert(accum, 0, 0)[bin_index]
        if verbose:
            print("index_wrt_class =", index_wrt_class)
        return bin_index, index_wrt_class

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
classDict = {}
if args.dataset == 'cifar10' or args.dataset == 'cifar100':
    if args.dataset == 'cifar100':
        classlbls = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']
        classes = ['shark', 'whale', 'baby']
    elif args.dataset == 'cifar10':
        classlbls = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        classes = ['dog', 'cat', 'ship']
    for lbl, idx in zip(classlbls, range(len(classlbls))):
      classDict[lbl] = idx

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
    ####################################################################################
    #################################### Evaluation ####################################
    ####################################################################################
    criterion = criterion1
    paths = [main_path + 'CrossEntropy.bin', main_path + 'LabelSmoothing.bin', main_path + 'Instance_LabelSmoothing_ce_logits.bin', 
             main_path + 'CETmpScaling.bin', main_path + 'LSTmpScaling.bin', main_path + 'ILSTmpScaling.bin']
    titles = ['CrossEntropy', 'LabelSmoothing', 'InstanceLabelSmoothing', 'CETmpScaling', 'LSTmpScaling', 'ILSTmpScaling']
    colors = ['r', 'g', 'b', 'c', 'y', 'm']
    fig = plt.figure(figsize=(20,10))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    for p, title, color in zip(paths, titles, colors):
        results_dict['model'] = title
        orig_net = get_new_model()
        states = torch.load(p)
        orig_net.load_state_dict(states)
        orig_net = orig_net.to(device)
        orig_net.eval()
        test_loss = []
        correct = 0; total = 0
        with torch.no_grad():
            ii = -1
            for batch_idx, (inputs, targets) in enumerate(testloader):
                ii += 1
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = orig_net(inputs)
                if ii == 0:
                    X = F.softmax(outputs, dim=-1).cpu().numpy()
                    y = targets.cpu().numpy()
                else:
                    X = np.concatenate((X, F.softmax(outputs, dim=-1).cpu().numpy()), axis=0)
                    y = np.concatenate((y, targets.cpu().numpy()), axis=0)
                loss = criterion(outputs, targets)
                test_loss.append(loss.cpu().item())
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'% (np.mean(test_loss), 100.*correct/total, correct, total))
        
        preds = np.argmax(X, axis = -1); confs = np.amax(X, axis = -1)
        bins = {0.1:[], 0.2:[], 0.3:[], 0.4:[], 0.5:[], 0.6:[], 0.7:[], 0.8:[], 0.9:[], 1:[]}; accs = {0.1:[], 0.2:[], 0.3:[], 0.4:[], 0.5:[], 0.6:[], 0.7:[], 0.8:[], 0.9:[], 1:[]}
        for pred, conf, yy in zip(preds, confs, y):
            for k in bins.keys():
                if conf <= k:
                    bins[k].append(conf)
                    if yy == pred:
                        accs[k].append(1)
                    else:
                        accs[k].append(0)
        xs = []; ys = []
        for k in bins.keys():
            xs.append(np.mean(bins[k]))
            ys.append(np.mean(accs[k]))
            ax1.annotate(str(len(accs[k])), (np.mean(bins[k]), np.mean(accs[k])), color = color)
        ax1.plot(xs, ys, label=title, color=color)
        if args.mutual_info:
            with open (title+'.pkl', 'rb') as fp:
                ys2 = pickle.load(fp)
            ax2.plot(list(range(1, len(ys2) * 3 + 1, 3)), ys2, color = color)
        ece = ECE(args.ece_bins); ece_score = ece.measure(X, y); acc = 100.*correct/total
        print('Testing LOS_LOSS:', np.mean(test_loss)); print('Testing ACCURACY:', acc); print('Testing ECE:', ece_score)
        results_dict['logloss'] = np.mean(test_loss); results_dict['acc'] = acc; results_dict['ece'] = ece_score
        results_df = results_df.append(results_dict, ignore_index=True)
        f = open(main_path + "evaluation_logs.txt", "a")
        f.write('###################################################################\n' + 'P:' + p + '\n')
        f.write('Mean Testing LOG_LOSS:' + str(np.mean(test_loss)) + '\n' + 'Testing ACCURACY:' + str(acc) + '\n' + 'Testing ECE:' + str(ece_score) + '\n')
        f.close()

    ax1.plot([0, 1], [0, 1], label='Perfect-Calibrated', color='k')
    ax1.set_xlabel('Confidence', fontsize=16)
    ax1.set_ylabel('Accuracy', fontsize=16)
    ax2.set_xlabel('Epochs', fontsize=16)
    ax2.set_ylabel('Mutual Information', fontsize=16)
    handles, labels = ax1.get_legend_handles_labels()
    lgd = ax1.legend(handles, labels)
    ax1.grid('on'); ax2.grid('on')
    if args.reliability or args.mutual_info:
        plt.savefig(main_path + 'evaluation_plots.png', bbox_inches='tight')

results_df.to_csv(main_path + 'results.csv', index = False)