# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 01:17:12 2019

@author: noron
"""

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
import string
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.patches
import os
import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation
from q4 import *
from torch.utils.data import DataLoader, TensorDataset


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
def getDataLoaders():
    

    trainset = torchvision.datasets.EMNIST('../data', split='balanced', train=True,
                download = False,
                transform=transforms.Compose([
                    transforms.ToTensor(),
               ]))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True, num_workers = 3)

    testset = torchvision.datasets.EMNIST(root='../data', split = 'balanced', train=False,
                                         download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                             shuffle=False, num_workers = 3)
    
    return trainloader, testloader

class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(),
                                   nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(),
                                   nn.MaxPool2d(stride=2, kernel_size=2)
                                   )
        self.conv2 = nn.Sequential(nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(),
                                   nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(),
                                   nn.MaxPool2d(stride=2, kernel_size=2)
                                   )
        self.fc1 = nn.Sequential(nn.Linear(7*7*32, 1024),
                                 nn.ReLU(),
                                 nn.Linear(1024, 47))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 7*7*32)
        x = self.fc1(x)
        return x
    
def train(trainloader, optimizer, criterion, epoch, net):
    net.train()
    total_loss = 0
    true = 0
    total_num = 0
    #for epoch in range(num_epochs):
    for data in trainloader:
        # Run the forward pass
        images = torch.autograd.Variable(data[0])
        labels = torch.autograd.Variable(data[1])
        
        outputs = net(images)
        
        
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += (loss.item())
        _, predicted = torch.max(outputs.data, 1)
        total_num= total_num+labels.size(0)
        true =true+ (predicted == labels).sum().item()
    accuracy =(true / total_num*100)
    mean_loss = total_loss/len(trainloader)
    return mean_loss, accuracy
    
 

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
    




            
    
def get_data(path, net):
    im1 = skimage.img_as_float(skimage.io.imread(path))
    bboxes, bw = findLetters(im1)

    plt.imshow(bw, cmap='gray')
    
    # find the rows using..RANSAC, counting, clustering, etc.
    # get centroid positions and width and height of box
    y_c = [(i[2] + i[0])/2 for i in bboxes]
    x_c = [(i[3] + i[1])/2 for i in bboxes]
    y_h = [(i[2] - i[0]) for i in bboxes]
    x_w = [(i[3] - i[1]) for i in bboxes]
    mean_height = sum(y_h)/len(y_h)
    positions = list(zip(y_c, x_c, y_h, x_w))
    positions = sorted(positions, key = lambda a: (a[0], a[1]))
    temp = positions[0][0]
    row = []
    rows = []
    for p in positions:
        if p[0] < temp + mean_height:
            
            row.append(p)
        else:
            row = sorted(row, key=lambda a: a[1])
            rows.append(row)
            row = [p]
            temp = p[0]
    row = sorted(row, key=lambda a: a[1])
    rows.append(row)
   
    
    rowsd = []
    for row in rows:
        rowd = []
        for y, x, h, w in row:
            
            im = bw[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)]
            
            h_pad, w_pad = 0, 0
            if h > w:
                h_pad = h/20
                w_pad = (h-w)/2+h_pad
            elif h < w:
                w_pad = w/20
                h_pad = (w-h)/2+w_pad
            im = np.pad(im, ((int(h_pad), int(h_pad)), (int(w_pad), int(w_pad))), 'constant', constant_values=(1, 1))
            
            im = skimage.transform.resize(im, (28, 28))
            im = skimage.morphology.erosion(im, np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
            rowd.append(np.transpose(im).flatten())
        rowsd.append(np.array(rowd))
    return rowsd
        
def detect(input_data, net):
    
    
    letters = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
         10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
         20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
         30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z', 36: 'a', 37: 'b', 38: 'd', 39: 'e',
         40: 'f', 41: 'g', 42: 'h', 43: 'n', 44: 'q', 45: 'r', 46: 't'}
    input_data1 = [transform(np.expand_dims(i, axis=2)).type(torch.float32) for i in input_data]
    x = torch.stack([i for i in input_data1]) 
    inputd = torch.autograd.Variable(x)

    
    y_pred = net(inputd)

    predicted = torch.max(y_pred, 1)[1]

    b = predicted.numpy()
    sen = ''
    for i in range(len(b)):
       
        sen += letters[b[i]]

    print(sen)
   






    

    

 
max_epoch = 10
trainloader, testloader = getDataLoaders()
net = Net1()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum = 0.9)

train_loss=[]
accuracy=[]
for epoch in range(max_epoch):
    tr_loss, tr_accuracy = train(trainloader, optimizer, criterion, epoch, net)
        
    print("itr: {:02d} \t loss: {:.2f} \t accuracy : {:.2f}".format(epoch+1, tr_loss, tr_accuracy))
    train_loss.append(tr_loss)
    accuracy.append(tr_accuracy)
    for img in os.listdir('../data_images/images'):
        img_path = os.path.join('../data_images/images',img)
        input_data = get_data(img_path, net)
        for row_data in input_data:
            detect(row_data, net)


    
plt.figure(1)
plt.plot(range(max_epoch), train_loss, 'k', label = 'training data')
plt.xlabel('Number of epochs')
plt.ylabel('Cross entropy loss')
plt.title('Average cross entropy loss vs number of epochs')
plt.legend()
plt.savefig('../Results/714_loss.png')
plt.show()
    

plt.figure(2)
plt.plot(range(max_epoch), accuracy, 'k', label = 'training data')
plt.xlabel('Number of epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy vs number of epochs')
plt.legend()
plt.savefig('../Results/714_acc.png')
plt.show()
    

    
