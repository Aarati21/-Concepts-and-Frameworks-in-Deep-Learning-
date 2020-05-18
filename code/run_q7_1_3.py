# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 21:10:26 2019

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
import scipy
from torch.utils.data import DataLoader, TensorDataset

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
    testloader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size=64, shuffle=False)

    return trainloader, testloader

class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.dropout =  nn.Dropout2d(p=0.5)
        self.pool = nn.MaxPool2d(stride=2, kernel_size=2, padding=0, dilation=1, ceil_mode=False)
        self.linear1 = nn.Linear(in_features=16*16*16, out_features=1024)
        self.linear2 = nn.Linear(in_features=1024, out_features=36)


    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.pool(self.dropout(self.conv2(out))))
       
        out = out.view(-1, 16*16*16)
       
        out = self.relu(self.linear1(out))
        
        out = self.linear2(out)
        return out
        
def train(trainloader, optimizer, criterion, epoch, net):
    net.train()
    total_loss = 0
    true = 0
    total_num = 0
    #for epoch in range(num_epochs):
    for x, y in trainloader:  
        inputs = torch.autograd.Variable(x)
        labels = torch.autograd.Variable(y)
        target = torch.max(labels, 1)[1]
        outputs = net(inputs)
        
        loss = criterion(outputs, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_num= total_num+labels.size(0)
        true =true+ (predicted == target.data).sum().item()
        
    acc =(true / total_num*100)
    
    
    return total_loss, acc
    
 

def test(testloader, criterion, net):
    net.eval()

    with torch.no_grad():
        true = 0
        total_num = 0
        for x, y in testloader:
            inputs = torch.autograd.Variable(x)
            labels = torch.autograd.Variable(y)
            target = torch.max(labels, 1)[1]
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_num= total_num +labels.size(0)
            true =true+ (predicted == target.data).sum().item()
        acc =(true / total_num*100)
        
    return acc
def main():
 
    max_epoch = 10
    trainloader, testloader = getDataLoaders()
    net = Net1()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    train_loss=[]
    accuracy=[]
    for epoch in range(max_epoch):
        tr_loss, tr_accuracy = train(trainloader, optimizer, criterion, epoch, net)
        
        print("itr: {:02d} \t loss: {:.2f} \t accuracy : {:.2f}".format(epoch+1, tr_loss, tr_accuracy))
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
    plt.savefig('../Results/713_loss.png')
    plt.show()
    

    plt.figure(2)
    plt.plot(range(max_epoch), accuracy, 'k', label = 'training data')
    plt.xlabel('Number of epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs number of epochs')
    plt.legend()
    plt.savefig('../Results/713_acc.png')
    plt.show()
    
    
if __name__=='__main__':

    main()    
