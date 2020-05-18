# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 12:07:26 2019

@author: noron
"""

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



def getDataLoaders():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)
    
    trainloader = torch.utils.data.DataLoader(trainset,
                                              shuffle=True)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                         download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, 
                                             shuffle=False)
    
    return trainloader, testloader

class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.c1 = nn.Conv2d(1, 10, kernel_size=(5,5), stride=(1,1))
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.relu = nn.ReLU()
        
        self.c2 = nn.Conv2d(10, 20, kernel_size=(5,5), stride=(1,1))
        self.dropout =  nn.Dropout2d(p=0.5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.fc1 = nn.Linear(in_features=320, out_features=50, bias=True)
        self.fc2 = nn.Linear(in_features=50, out_features=10, bias=True)
        pass
            
    def forward(self, x):
        f1 = self.c1(x)
        out=self.pool1(f1)
        out=self.relu(out)
        
        f2 = self.c2(out)
        out=self.dropout(f2)
        out=self.pool2(out)
        out=self.relu(out)
        
        out = out.reshape(out.size(0), -1)
        
        out=self.fc1(out)
        out=self.dropout(self.relu(out))
        out = out.reshape(out.size(0), -1)
        out = self.fc2(out)
        
        return f1, f2, out
    
def train(trainloader, optimizer, criterion, epoch, net):
    net.train()
    total_loss = 0
    true = 0
    total_num = 0
    #for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(trainloader):
        # Run the forward pass
        
        _,_,outputs = net(images)
        
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += (loss.item())
        _, predicted = torch.max(outputs.data, 1)
        total_num= total_num+labels.size(0)
        true =true+ (predicted == labels).sum().item()
    accuracy =(true / total_num*100)
    
    return total_loss, accuracy
    
 

def test(testloader, criterion, net):
    net.eval()
    total_loss = 0
    with torch.no_grad():
        true = 0
        total_num = 0
        for images, labels in testloader:
            #images = images.reshape(-1, 28*28)
           
            _,_,outputs = net(images)
           
            loss = criterion(outputs, labels)
            total_loss += (loss.item())
    
            _, predicted = torch.max(outputs.data, 1)
            total_num= total_num + labels.size(0)
            true = true+ (predicted == labels).sum().item()
    accuracy=(true / total_num*100)
    mean_loss = total_loss/len(testloader)
    return accuracy, mean_loss
    
    


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

    te_acc, te_loss = test(testloader, criterion, net)
    print("test_loss: {:.2f} \t test_accuracy : {:.2f}".format(te_loss, te_acc))
    
    plt.figure(1)
    plt.plot(range(max_epoch), train_loss, 'k', label = 'training data')
    plt.xlabel('Number of epochs')
    plt.ylabel('Cross entropy loss')
    plt.title('Average cross entropy loss vs number of epochs')
    plt.legend()
    plt.savefig('../Results/712_loss.png')
    plt.show()
    

    plt.figure(2)
    plt.plot(range(max_epoch), accuracy, 'k', label = 'training data')
    plt.xlabel('Number of epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs number of epochs')
    plt.legend()
    plt.savefig('../Results/712_acc.png')
    plt.show()
    
    
if __name__=='__main__':

    main()    
