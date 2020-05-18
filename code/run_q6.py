import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt
from skimage.measure import compare_psnr as psnr
import numpy.linalg as l
train_data = scipy.io.loadmat('../data1/nist36_train_set1.mat')
#take test set here
valid_data = scipy.io.loadmat('../data1/nist36_test_set1.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['test_data']


# do PCA
mean_data = np.mean(train_x, axis = 0)
train_x -= mean_data
U, S, V = np.linalg.svd(train_x)
X = V[:32, :]

# rebuild a low-rank version
lrank = train_x.dot(X.T)

# rebuild it
recon = lrank.dot(X)

recon += mean_data
train_x += mean_data
psn = 0
for i in range(len(recon)):
    psn += psnr(train_x[i], recon[i])
print('PSNR Value is', psn/len(recon))

ind1 = [3, 140, 355, 775, 1625 ] 
ind2 = [43, 101, 360, 785, 1600]


# build valid dataset
valid_mean = np.mean(valid_x, axis = 0)
valid_x -= valid_mean
recon_valid_rank = valid_x.dot(X.T)
recon_valid = recon_valid_rank.dot(X)
recon_valid += valid_mean
valid_x +=valid_mean


total = []
for pred,gt in zip(recon_valid,valid_x):
    total.append(psnr(gt,pred))
print(np.array(total).mean())

for i in range(len(ind1)):
    plt.subplot(1,4, 1)
    plt.imshow(valid_x[ind1[i]].reshape(32,32).T)
    plt.subplot(1,4, 2)
    plt.imshow(recon_valid[ind1[i]].reshape(32,32).T)

    plt.subplot(1,4, 3)
    plt.imshow(valid_x[ind2[i]].reshape(32,32).T)
    plt.subplot(1,4, 4)
    plt.imshow(recon_valid[ind2[i]].reshape(32,32).T)  
              
    plt.show()
