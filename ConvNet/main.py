'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import socket
hostname = socket.gethostname()

from VGG_Cifar10 import *
from utils_device import *
from utils import *



parser = argparse.ArgumentParser(description = 'PyTorch CIFAR10 Training')
parser.add_argument('-lr', default = 5e-3, type = float, help = 'learning rate')
parser.add_argument('-lr_decay', default = 15, type = float, help = 'learning rate')
parser.add_argument('--batch_size', '-bs', default = 500, type = int, help = 'batch_size')
parser.add_argument('-it', default = 1, type = int, help = 'number of iterations before update')
parser.add_argument('--epochs', '-e', default = 700, type = int, help = '# of epochs')
parser.add_argument('--gpus', '-gpus', '--gpu' , '-gpu', default = '0', help = 'gpus used for training - e.g 0,1,3')
parser.add_argument('--resume', '-r', action = 'store_true', help = 'resume from checkpoint')
parser.add_argument('--weight-decay', '-wd', default = 1e-2, type = float, metavar = 'W', help = 'weight decay (default: 5e-4)')
parser.add_argument('-lr_decay_gamma', default = 50, type = float, help = 'learning rate decay gamma')
args = parser.parse_args()

if args.gpus != 'None' :
	args.gpus = [int(i) for i in args.gpus.split(',')]

#torch.manual_seed(0)

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

fichier = open(str(hostname) + str(args.gpus) +'_resultat.csv', 'w')
print('lr = '+ str(args.lr)+ ' | bs = ' + str(args.batch_size) + ' | lr_decay_epoch = ' + str(args.lr_decay)+ ' | lr_decay_gamma = ' + str(args.lr_decay_gamma) + ' | test accuracy, train accuracy, train loss ',file=fichier)
fichier.flush()

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=200, shuffle=False)

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
trainloader_tested = torch.utils.data.DataLoader(trainset, batch_size=200, shuffle=True)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

if args.gpus != 'None' :
	device = torch.device("cuda:" + str(args.gpus[0]) if torch.cuda.is_available() else "cpu")
else : 
    device = torch.device("cpu")


# Model
print('==> Building model..')

net = VGG_Cifar10()

#net = convert_model(net)
if args.gpus != 'None' :
    if args.gpus and len(args.gpus) > 1 :
        net = nn.DataParallel(net, args.gpus)

print(net)

net = net.to(device)

#net.apply(init_weights)

#print("Running for 750 epochs!")
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('./checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']


criterion = nn.CrossEntropyLoss()
optimizer = Adam_with_device(net.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay, gamma=args.lr_decay_gamma)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4, eta_min=1, last_epoch=-1)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    print(scheduler.get_lr())
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        # This loop is for Binary parameters having 'org' attribute
        for p in list(net.parameters()): # blocking weights with org value greater than a threshold by setting grad to 0
            if hasattr(p,'org'):
                p.data.copy_(p.org)
      
        optimizer.step()
       
        # This loop is only for Binary parameters as they have 'org' attribute
        for p in list(net.parameters()):  # updating the org attribute
            if hasattr(p,'org'):
                p.org.copy_(p.data)

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Train loss during training: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    train_loss = 0
    correct_test = 0
    correct_train = 0
    total_train = 0
    total_test = 0
    with torch.no_grad():

        for batch_idx, (inputs, targets) in enumerate(trainloader_tested):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += targets.size(0)
            correct_train += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader_tested), 'Train loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct_train/total_train, correct_train, total_train))

        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total_test += targets.size(0)
            correct_test += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Test loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct_test/total_test, correct_test, total_test))

    acc_train = 100.*correct_train/total_train
    acc_test = 100.*correct_test/total_test
    # Append accuracy
    print('%.3f' % acc_test + ', %.3f' % acc_train + ', %.5f' % train_loss, file=fichier)
    fichier.flush()
    
    # Save checkpoint.
    if acc_test > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc_train': acc_train,
            'acc_test': acc_test,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc_test


for epoch in range(start_epoch, start_epoch+args.epochs):
    train(epoch)
    test(epoch)
    scheduler.step()
