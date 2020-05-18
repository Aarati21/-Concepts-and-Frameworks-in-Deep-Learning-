# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 19:52:34 2019

@author: noron
"""
import scipy.io
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt 
from torch.utils.data import DataLoader, TensorDataset

device = torch.device('cpu')



train_data = scipy.io.loadmat('../data1/nist36_train_set1.mat')
test_data = scipy.io.loadmat('../data1/nist36_test_set1.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
test_x, test_y = test_data['test_data'], test_data['test_labels']


tensortr_x = torch.stack([torch.Tensor(i) for i in train_x]) 
tensortr_y = torch.stack([torch.Tensor(i) for i in train_y])
tensorte_x = torch.stack([torch.Tensor(i) for i in test_x]) 
tensorte_y = torch.stack([torch.Tensor(i) for i in test_y])

train_dataset = TensorDataset(tensortr_x, tensortr_y)
test_dataset = TensorDataset(tensorte_x, tensorte_y)

max_iters = 5
batch_size = 32
learning_rate = 0.01
input_dimension = train_x.shape[1]
hidden_dimension = 64
output_dimension = train_y.shape[1]

train_dataloader = torch.utils.data.DataLoader(dataset = train_dataset, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size=batch_size, shuffle=False)


class SimpleNeuralNet(nn.Module):
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        super(SimpleNeuralNet, self).__init__()
        self.l1=nn.Linear(input_nodes, hidden_nodes)
        self.sigmoid=nn.Sigmoid()
        self.l2=nn.Linear(hidden_nodes, output_nodes)
        

    def forward(self, x):
        output=self.l1(x)
        output = self.sigmoid(output)
        output=self.l2(output)
        return output
    
model = SimpleNeuralNet(input_dimension, hidden_dimension, output_dimension)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum = 0.9)
train_loss = []
train_acc = []   
for i in range(max_iters):
    total_loss = 0
    
    true = 0
    for x, y in train_dataloader:  
        inputs = torch.autograd.Variable(x)
        
        labels = torch.autograd.Variable(y)
        target = torch.max(labels, 1)[1]
        outputs = model(inputs)
        loss = criterion(outputs, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        
        true =true+ (predicted == target.data).sum().item()
        
    acc = true/len(train_y)
    train_loss.append(total_loss)
    train_acc.append(acc)

    if i % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(i, total_loss, acc*100))
        
plt.figure(1)
plt.plot(range(max_iters), train_loss, 'k', label = 'training data')
plt.xlabel('Number of epochs')
plt.ylabel('Cross entropy loss')
plt.title('Average cross entropy loss vs number of epochs')
plt.legend()
plt.savefig('../Results/711_loss')
plt.show()


plt.figure(2)
plt.plot(range(max_iters), train_acc, 'k', label = 'training data')
plt.xlabel('Number of epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy vs number of epochs')
plt.legend()
plt.savefig('../Results/711_acc')
plt.show()


with torch.no_grad():
    true = 0
    for x, y in test_dataloader:
        print(x.shape)
        inputs = torch.autograd.Variable(x)
        labels = torch.autograd.Variable(y)
        target = torch.max(labels, 1)[1]
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
       
        true =true+ (predicted == target.data).sum().item()

    print('Test Accuracy is: {} %'.format(100 * true / len(test_y)))
   