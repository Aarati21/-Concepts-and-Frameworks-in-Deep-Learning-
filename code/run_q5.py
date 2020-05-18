import numpy as np
import scipy.io
from nn import *
from collections import Counter

train_data = scipy.io.loadmat('../data1/nist36_train_set1.mat')
valid_data = scipy.io.loadmat('../data1/nist36_valid_set1.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

max_iters = 100
# pick a batch size, learning rate
batch_size = 36 
learning_rate =  3e-5
hidden_size = 32
lr_rate = 20
batches = get_random_batches(train_x,np.ones((train_x.shape[0],1)),batch_size)
batch_num = len(batches)

params = Counter()

# initialize layers here
initialize_weights(1024, hidden_size, params, 'layer1')
initialize_weights(hidden_size, hidden_size, params, 'hidden')
initialize_weights(hidden_size, hidden_size, params, 'hidden2')
initialize_weights(hidden_size, 1024, params, 'output')

# populate the parameters dictionary with zero-initialized momentum accumulators, one for each parameter.
keys = []
for i in params.keys():
    keys.append(i)
for i in keys:
    params['m_'+i] = np.zeros(params[i].shape)

train_loss = []
# should look like your previous training loops
for itr in range(max_iters):
    total_loss = 0
    for xb,_ in batches:
        pass
        # training loop can be exactly the same as q2!
        # your loss is now squared error
        # delta is the d/dx of (x-y)^2
        # to implement momentum
        #   just use 'm_'+name variables
        #   to keep a saved value over timestamps
        #   params is a Counter(), which returns a 0 if an element is missing
        #   so you should be able to write your loop without any special conditions
        h1 = forward(xb, params, 'layer1', relu)
        h2 = forward(h1, params, 'hidden', relu)
        h3 = forward(h2, params, 'hidden2', relu)
        out = forward(h3, params, 'output', sigmoid)
        
        # loss
        loss = np.sum((out - xb)**2)
        total_loss += loss
        
        # backward
        d1 = 2*(out - xb)
        d2 = backwards(d1, params, 'output', sigmoid_deriv)
        d3 = backwards(d2, params, 'hidden2', relu_deriv)
        d4 = backwards(d3, params, 'hidden', relu_deriv)
        backwards(d4, params, 'layer1', relu_deriv)
        
        for i in params.keys():
            if '_' in i:
                continue
            params['m_'+i] = 0.9*params['m_'+i] - learning_rate*params['grad_'+i]
            params[i] += params['m_'+i]
            
            
        # loss
    train_loss.append(total_loss)   
    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f}".format(itr,total_loss))
    if itr % lr_rate == lr_rate-1:
        learning_rate *= 0.9

#5.2
import matplotlib.pyplot as plt
plt.figure(1)
plt.plot(range(max_iters), train_loss, 'k', label = 'training data')

plt.xlabel('Number of epochs')
plt.ylabel('Training loss')
plt.title('Total loss vs number of epochs')
plt.legend()
plt.savefig('../Results/52_loss.png')  
plt.show()
     
        
# visualize some results
# Q5.3.1

h1 = forward(valid_x,params,'layer1',relu)
h2 = forward(h1,params,'hidden',relu)
h3 = forward(h2,params,'hidden2',relu)
out = forward(h3,params,'output',sigmoid)


ind1 = [3, 273, 792, 1550, 3250 ]
ind2 = [73, 203, 720, 1570, 3201]
for i in range(len(ind1)):
    plt.subplot(1,4, 1)
    plt.imshow(valid_x[ind1[i]].reshape(32,32).T)
    plt.subplot(1,4, 2)
    plt.imshow(out[ind1[i]].reshape(32,32).T)

    plt.subplot(1,4, 3)
    plt.imshow(valid_x[ind2[i]].reshape(32,32).T)
    plt.subplot(1,4, 4)
    plt.imshow(out[ind2[i]].reshape(32,32).T)  
              
    plt.show()


from skimage.measure import compare_psnr as psnr
# evaluate PSNR
# Q5.3.2
psn = 0
h1 = forward(valid_x,params,'layer1',relu)
h2 = forward(h1,params,'hidden',relu)
h3 = forward(h2,params,'hidden2',relu)
out = forward(h3,params,'output',sigmoid)

for i in range(len(valid_x)):
    psn += psnr(valid_x[i], out[i])
print('PSNR Value is', psn/len(valid_x))