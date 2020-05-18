# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 21:30:36 2019

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
from torchvision.datasets import ImageFolder
batch_size = 32
def getDataLoaders():
    transform = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    trainset = ImageFolder(root='../data/oxford-flowers17/train', transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, num_workers = 2,
                                              shuffle=True)
    
    val_transform = transforms.Compose([
            transforms.Scale(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    valset = ImageFolder(root = '../data/oxford-flowers17/val', transform=val_transform)
    valloader = torch.utils.data.DataLoader(valset, batch_size = batch_size, num_workers = 2,
                                              shuffle=False)

    
    testset = ImageFolder(root = '../data/oxford-flowers17/test', transform= val_transform)
    testloader = torch.utils.data.DataLoader(testset,batch_size=batch_size, num_workers = 2,
                         shuffle=False)
    
    return trainloader, valloader, testloader

class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.c1 = nn.Sequential(nn.Conv2d(3, 10, kernel_size=7, stride=1, padding=3),
                                   nn.ReLU(),
                                   nn.MaxPool2d(stride=2, kernel_size=2)
                                   )
        self.c2 = nn.Sequential(nn.Conv2d(10, 20, kernel_size=5, stride=1, padding=2),
                                   nn.ReLU(),
                                   nn.MaxPool2d(stride=2, kernel_size=2)
                                   )
        self.c3 = nn.Sequential(nn.Conv2d(20, 64, kernel_size=5, stride=1, padding=2),
                                   nn.ReLU(),
                                   nn.MaxPool2d(stride=2, kernel_size=2)
                                   )
        self.fc1 = nn.Sequential(nn.Linear(28*28*64, 32),
                                 nn.ReLU()
                                 )
        self.fc2 = nn.Sequential(nn.Linear(32, 17))

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = x.view(-1, 28*28*64)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
        
            
def train(trainloader, optimizer, criterion, epoch, net):
    net.train()
    total_loss = 0
    true = 0
    total_num = 0
    #for epoch in range(num_epochs):
    for i, data in enumerate(trainloader):
        # Run the forward pass
        inputs = torch.autograd.Variable(data[0].type(torch.FloatTensor))
        labels = torch.autograd.Variable(data[1].type(torch.FloatTensor).long())
        outputs = net(inputs)
        
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
    
def valid(valloader, criterion, net):
    net.eval()
    total_loss = 0
    true = 0
    total_num = 0
    for i, data in enumerate(valloader):
            #images = images.reshape(-1, 28*28)
        inputs = torch.autograd.Variable(data[0])
        labels = torch.autograd.Variable(data[1])
        outputs = net(inputs)
           
        loss = criterion(outputs, labels)
        total_loss += (loss.item())
    
        _, predicted = torch.max(outputs.data, 1)
        total_num= total_num + labels.size(0)
        true = true+ (predicted == labels).sum().item()
    accuracy=(true / total_num*100)
   
    return  total_loss, accuracy

def test(testloader, criterion, net):
    net.eval()
    total_loss = 0
    with torch.no_grad():
        true = 0
        total_num = 0
        for i, data in enumerate(testloader):
            #images = images.reshape(-1, 28*28)
            inputs = torch.autograd.Variable(data[0].type(torch.FloatTensor))
            labels = torch.autograd.Variable(data[1].type(torch.FloatTensor).long())
            outputs = net(inputs)
           
            loss = criterion(outputs, labels)
            total_loss += (loss.item())
    
            _, predicted = torch.max(outputs.data, 1)
            total_num= total_num + labels.size(0)
            true = true+ (predicted == labels).sum().item()
    accuracy=(true / total_num*100)
    
    return accuracy, total_loss
    
    
    


def main():
 
    max_epoch1 =100
    max_epoch2 =50
    learning_rate1 = 0.001
    learning_rate2 = 0.0001
    trainloader, valloader, testloader = getDataLoaders()
    net = Net1()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate1)
    train_loss=[]
    accuracy=[]
    print('Stage1')
    for epoch in range(max_epoch1):
        tr_loss, tr_accuracy = train(trainloader, optimizer, criterion, epoch, net)
        
        print("itr: {:02d} \t loss: {:.2f} \t accuracy : {:.2f}".format(epoch+1, tr_loss, tr_accuracy))
        train_loss.append(tr_loss)
        accuracy.append(tr_accuracy)
        val_loss, val_accuracy = train(trainloader, optimizer, criterion, epoch, net)
        
    print('Stage2')
    for epoch in range(max_epoch2):
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate2)
        tr_loss, tr_accuracy = train(trainloader, optimizer, criterion, epoch, net)
        
        print("itr: {:02d} \t loss: {:.2f} \t accuracy : {:.2f}".format(epoch+1, tr_loss, tr_accuracy))
        train_loss.append(tr_loss)
        accuracy.append(tr_accuracy)
        val_loss, val_accuracy = train(trainloader, optimizer, criterion, epoch, net)
        
    te_acc, te_loss = test(testloader, criterion, net)
    print("val_loss: {:.2f} \t val_accuracy : {:.2f}".format(te_loss, te_acc))
    
    plt.figure(1)
    plt.plot(range(60), train_loss, 'k', label = 'training data')
    plt.xlabel('Number of epochs')
    plt.ylabel('Cross entropy loss')
    plt.title('Average cross entropy loss vs number of epochs')
    plt.legend()
    plt.savefig('../Results/72fine_loss.png')
    plt.show()
    

    plt.figure(2)
    plt.plot(range(60), accuracy, 'k', label = 'training data')
    plt.xlabel('Number of epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs number of epochs')
    plt.legend()
    plt.savefig('../Results/72fine_acc.png')
    plt.show()
    
    
if __name__=='__main__':

    main()    






