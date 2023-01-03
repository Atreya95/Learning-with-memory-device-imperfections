from __future__ import print_function
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from binarized_modules import *

import sys
import os
os.environ['CUDA_LAUNCH_BLOCKING']="1"

from utils import *

import socket
hostname = socket.gethostname()

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST training with device imperfections')
parser.add_argument('-it', default=1, type=int, help='number of iterations before update')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('-lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--gpus', default='0',
                    help='gpus used for training - e.g 0,1,3')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--wd', type=float, default=0, metavar='LR',
                    help='learning rate (default: 1E-7)')

args = parser.parse_args()

if args.gpus != 'None' :
	args.gpus = [int(i) for i in args.gpus.split(',')]

fichier = open(str(hostname) + str(args.gpus) +'_binary_mnist.csv', 'w')

kwargs = {'num_workers': 1, 'pin_memory': True}


train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./mnist_dataset', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor()])),
    batch_size = 1000, shuffle = True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./mnist_dataset', train = False, transform = transforms.Compose([
                       transforms.ToTensor()])),
    batch_size = 1000, shuffle = True)


if args.gpus != 'None' :
	device = torch.device("cuda:" + str(args.gpus[0]) if torch.cuda.is_available() else "cpu")
else : 
    device = torch.device("cpu")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.infl_ratio=8
        self.fc1 = BinarizeLinear(784, 3000, bias=True)
        self.bn1 = nn.BatchNorm1d(3000)
        self.htanh1 = nn.Hardtanh()
        self.fc2 = BinarizeLinear(3000, 10, bias=True)
        self.bn2 = nn.BatchNorm1d(10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.htanh1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        return x

model = Net()
if args.gpus != 'None' :
    if args.gpus and len(args.gpus) > 1 :
        model = nn.DataParallel(model, args.gpus)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = Adam_with_device(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

def train(epoch):
    model.train()
    correct_train = 0
    total_train = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        for p in list(model.parameters()):
            if hasattr(p,'org'):
                p.data.copy_(p.org)
        optimizer.step()
        for p in list(model.parameters()):
            if hasattr(p,'org'):
                 p.org.copy_(p.data)#.clamp_(-1,1)

        _, predicted = output.max(1)
        total_train += target.size(0)
        correct_train += predicted.eq(target).sum().item()

        if batch_idx % args.log_interval == 0:
            print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy:({:.1f}%)'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), 100.*correct_train/total_train), end="\r")
            sys.stdout.write("\033[K")
            

def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data, target = Variable(data), Variable(target)
            output = model(data)
            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader)
    test_acc = float(100 * correct) / float(len(test_loader.dataset))
    print('Train Epoch: {} | Test set -> loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)'.format(epoch,
        test_loss, correct, len(test_loader.dataset), test_acc))
    print('%.3f' % (test_acc),file=fichier)
    fichier.flush()
    return test_acc

test_accuracy = []
for epoch in range(1, args.epochs + 1):
    train(epoch)
    test()
    test_accuracy.append(test())
    scheduler.step()


np.savetxt("BNN_with_device_simulation_test_accuracy.txt", test_accuracy)
