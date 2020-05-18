# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 21:23:18 2019

@author: noron
"""

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim 
import numpy as np
import scipy.io
from torch.utils.data import DataLoader, TensorDataset
device = torch.device('cpu')
torch.manual_seed(123)
def getDataLoaders():
    train_data = scipy.io.loadmat('../data1/nist36_train_set1.mat')
    test_data = scipy.io.loadmat('../data1/nist36_test_set1.mat')
    train_x, train_y = train_data['train_data'], train_data['train_labels']
    test_x, test_y = test_data['test_data'], test_data['test_labels']
    train_x = np.array([np.reshape(i, (32, 32)) for i in train_x])
    test_x = np.array([np.reshape(i, (32, 32)) for i in test_x])

    tensortr_x = (torch.stack([torch.Tensor(i) for i in train_x])).unsqueeze(1) 
    tensortr_y = torch.stack([torch.LongTensor(i) for i in train_y])
    tensorte_x = (torch.stack([torch.Tensor(i) for i in test_x])).unsqueeze(1) 
    tensorte_y = torch.stack([torch.LongTensor(i) for i in test_y])

    train_dataset = TensorDataset(tensortr_x, tensortr_y)
    test_dataset = TensorDataset(tensorte_x, tensorte_y)
    
    trainloader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(dataset = test_dataset, shuffle=False)

    return trainloader, testloader



class SkipConnection(torch.nn.Module):

    def __init__(self, channels):
        
        super(SkipConnection, self).__init__()
        self.conv_1 = torch.nn.Conv2d(in_channels=channels[0],
                                      out_channels=channels[1],
                                      kernel_size=(3, 3),
                                      stride=(2, 2),
                                      padding=1)
        self.conv_1_bn = torch.nn.BatchNorm2d(channels[1])
                                    
        self.conv_2 = torch.nn.Conv2d(in_channels=channels[1],
                                      out_channels=channels[2],
                                      kernel_size=(1, 1),
                                      stride=(1, 1),
                                      padding=0)   
        self.conv_2_bn = torch.nn.BatchNorm2d(channels[2])

        self.conv_shortcut_1 = torch.nn.Conv2d(in_channels=channels[0],
                                               out_channels=channels[2],
                                               kernel_size=(1, 1),
                                               stride=(2, 2),
                                               padding=0)   
        self.conv_shortcut_1_bn = torch.nn.BatchNorm2d(channels[2])

    def forward(self, x):
        shortcut = x
        
        out = self.conv_1(x)
        out = self.conv_1_bn(out)
        out = F.relu(out)

        out = self.conv_2(out)
        out = self.conv_2_bn(out)
        
        # match up dimensions using a linear function (no relu)
        shortcut = self.conv_shortcut_1(shortcut)
        shortcut = self.conv_shortcut_1_bn(shortcut)
        
        out1 = torch.cat((out, shortcut), 1)
        out = F.relu(out1)

        return out
    
class ConvNet(torch.nn.Module):

    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        
        self.residual_block_1 = SkipConnection(channels=[1, 5, 10])
        self.residual_block_2 = SkipConnection(channels=[20, 40, 60])
    
        self.linear_1 = torch.nn.Linear(120*8*8, num_classes)

        
    def forward(self, x):

        out1 = self.residual_block_1.forward(x)
        
        out2 = self.residual_block_2.forward(out1)
        
        logits = self.linear_1(out2.view(-1, 120*8*8))
        
        probas = F.softmax(logits, dim=1)
        return out1, out2,  probas
        
def train(trainloader, criterion, optimizer, epoch, net):
    net.train()
    total_loss = 0
    true = 0
    total_num = 0
    #for epoch in range(num_epochs):
    for x, y in trainloader:  
        inputs = torch.autograd.Variable(x)
        labels = torch.autograd.Variable(y)
        target = torch.max(labels, 1)[1]
        _, _, outputs = net(inputs)
        loss = criterion(outputs, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        _, predicted = torch.max(outputs.data, 1)
        total_num= total_num+labels.size(0)
        true =true+ (predicted == target.data).sum().item()
        
    acc =(true / total_num*100)
    mean_loss = total_loss/len(trainloader)
    
    return mean_loss, acc
    
 

def test(testloader, criterion, net):
    net.eval()

    with torch.no_grad():
        true = 0
        total_num = 0
        for x, y in testloader:
            
            inputs = torch.autograd.Variable(x)
            labels = torch.autograd.Variable(y)
            target = torch.max(labels, 1)[1]
            _, _, outputs = net(inputs)
            
            _, predicted = torch.max(outputs.data, 1)
            
            total_num= total_num +labels.size(0)
            true =true+ (predicted == target.data).sum().item()
        acc =(true / total_num*100)
    
    return acc


def vizFeatureMaps(testloader, net):

    """ Visualize weights of the convolution layers

        Arguments:
            testloader: Dataloader for test dataset
            net: instance of the CNN class

        TO-DO:
        1. Pass one image through the network and get its conv1 and conv2 features
        2. conv1: Get the features after conv1. There'll be 10 24x24 feature maps. Create a subplot and visualize them
        3. conv2: Get the features after conv2. There'll be 20 8x8 feature maps. Create a subplot and visualize them
    """
    with torch.no_grad():
        for iter, (images, labels) in enumerate(testloader):
            f1, f2, output=net(images)
            break
    plt.figure(figsize=(15,10))
    plt.tight_layout()
    print('The feature maps after conv 1 are visualized as:')
    for i in range(20):
    
        plt.subplot(4,5,i+1)
        plt.imshow(f1[0,i,:,:])
        plt.axis("off")
        plt.title("%d"%(i+1))
    plt.show()
    plt.figure(figsize=(24,20))
    plt.tight_layout()
    print('The feature maps after conv 2 are visualized as:')
    for i in range(120):
    
        plt.subplot(12,10,i+1)
        plt.imshow(f2[0,i,:,:])
        plt.axis("off")
        plt.title("%d"%(i+1))
    plt.show()
def main():
 
    max_epoch = 20
    trainloader, testloader = getDataLoaders()
    net = ConvNet(num_classes=36)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    train_loss=[]
    accuracy=[]
    for epoch in range(max_epoch):
        tr_loss, tr_accuracy = train(trainloader, criterion, optimizer, epoch, net)
        
        print("itr: {:02d}  \t accuracy : {:.2f}".format(epoch+1, tr_accuracy))
        train_loss.append(tr_loss)
        accuracy.append(tr_accuracy)

    te_acc = test(testloader, criterion, net)
   
    print(" test_accuracy : {:.2f}".format(te_acc))
    
    plt.figure(1)
    plt.plot(range(max_epoch), train_loss, 'k', label = 'training data')
    plt.xlabel('Number of epochs')
    plt.ylabel('Cross entropy loss')
    plt.title('Average cross entropy loss vs number of epochs')
    plt.legend()
    plt.savefig('../Results/7151_loss.png')
    plt.show()
    

    plt.figure(2)
    plt.plot(range(max_epoch), accuracy, 'k', label = 'training data')
    plt.xlabel('Number of epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs number of epochs')
    plt.legend()
    plt.savefig('../Results/7151_acc.png')
    plt.show()
    
    
    vizFeatureMaps(testloader, net)
    
if __name__=='__main__':

    main()    
