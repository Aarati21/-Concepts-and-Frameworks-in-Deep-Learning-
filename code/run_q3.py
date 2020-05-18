import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt
train_data = scipy.io.loadmat('../data2/nist36_train_set2.mat')
valid_data = scipy.io.loadmat('../data2/nist36_valid_set2.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']

max_iters = 50
# pick a batch size, learning rate
batch_size = 32
learning_rate = 0.01
hidden_size = 64

batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)

params = {}

# initialize layers here
in_size = len(train_x[0])
out_size = len(train_y[0])
initialize_weights(in_size, hidden_size, params, 'layer1')
initialize_weights(hidden_size, out_size, params, 'output')
train_loss = []
valid_loss = []
train_acc = []
valid_accs = []
# with default settings, you should get loss < 150 and accuracy > 80%
for itr in range(max_iters):
    total_loss = 0.0
    total_acc = 0.0
    for xb,yb in batches:
        
        # forward
        h = forward(xb, params, 'layer1', sigmoid)
        out = forward(h, params, 'output', softmax)
        
        # loss
        loss, accuracy = compute_loss_and_acc(yb, out)
        
        # be sure to add loss and accuracy to epoch totals 
        total_loss += loss
        total_acc += accuracy
        
        # backward
        delta = out - yb
        d_out = backwards(delta, params, 'output', linear_deriv)
        d_in = backwards(d_out, params, 'layer1', sigmoid_deriv)
        
        # apply gradient
        params['W'+'layer1'] -= learning_rate * params['grad_W' + 'layer1']
        params['W'+'output'] -= learning_rate * params['grad_W' + 'output']
        params['b'+'layer1'] -= learning_rate * params['grad_b' + 'layer1']
        params['b'+'output'] -= learning_rate * params['grad_b' + 'output']
        
    total_acc = total_acc/len(batches)
    train_loss.append(total_loss/train_x.shape[0])
    train_acc.append(total_acc)
    
    
    v_h1 = forward(valid_x, params, 'layer1', sigmoid)
    v_out = forward(v_h1, params, 'output', softmax)
    v_loss, v_acc = compute_loss_and_acc(valid_y, v_out)
    
    valid_accs.append(v_acc)
    valid_loss.append(float(v_loss)/valid_x.shape[0])
    
    
    #if itr % 100 == 0:
        #print("itr: {:02d} \t loss: {:.2f} \t acc : {:.8f}".format(itr,total_loss,avg_acc))
        
    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,total_acc))
# run on validation set and report accuracy! should be above 75%
valid_acc = valid_accs[-1]
print('Validation accuracy: ',valid_acc)

plt.figure(1)
plt.plot(range(max_iters), train_loss, 'k', label = 'training data')
plt.plot(range(max_iters), valid_loss, 'b', label = 'validation data')
plt.xlabel('Number of epochs')
plt.ylabel('Cross entropy loss')
plt.title('Average cross entropy loss vs number of epochs')
plt.legend()
plt.savefig('../Results2/315_loss_lr.png')
plt.show()


plt.figure(2)
plt.plot(range(max_iters), train_acc, 'k', label = 'training data')
plt.plot(range(max_iters), valid_accs, 'b', label = 'validation data')
plt.xlabel('Number of epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy vs number of epochs')
plt.legend()
plt.savefig('../Results2/315_acc_lr.png')
plt.show()

test_data = scipy.io.loadmat('../data2/nist36_test_set2.mat')
test_x, test_y = test_data['test_data'], test_data['test_labels']
h = forward(test_x, params, 'layer1', sigmoid)
out = forward(h, params, 'output', softmax)
test_loss, test_accuracy = compute_loss_and_acc(test_y, out)
print('Test accuracy: ',test_accuracy)


if False: # view the data
    for crop in xb:
        import matplotlib.pyplot as plt
        plt.imshow(crop.reshape(32,32).T)
        plt.show()
import pickle
saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Q3.1.3
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
with open('q3_weights.pickle', 'rb') as handle:
   saved_params = pickle.load(handle)



# Learned weights
weights = saved_params['Wlayer1']
fig = plt.figure(3)
mapme = ImageGrid(fig, 111, (8,8))
for i in range(hidden_size):
    #mapme[i].imshow(saved_params['Wlayer1'][:, i].reshape(hidden_size, hidden_size))
    mapme[i].imshow(saved_params['Wlayer1'][:, i].reshape(32, 32))
plt.savefig('../Results2/F_W.png')
plt.show()


initialize_weights(in_size, hidden_size, saved_params, 'initial')
fig = plt.figure(4)
mapme = ImageGrid(fig, 111, (8,8))
for i in range(hidden_size):
    mapme[i].imshow(saved_params['Winitial'][:, i].reshape(32, 32))
plt.savefig('../Results2/O_W.png')
plt.show()







# Q3.1.4
confusion_matrix = np.zeros((train_y.shape[1],train_y.shape[1]))

def cm(x, y):
    h = forward(x, params, 'layer1', sigmoid)
    out = forward(h, params, 'output', softmax)
    y_label = np.argmax(y, axis = 1)
    y_pred = np.argmax(out, axis = 1)
    for i in range(len(y_label)):
        confusion_matrix[y_label[i], y_pred[i]]+=1
    return confusion_matrix
 
confusion_matrix = cm(test_x, test_y)
import string
plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.imsave('../Results2/cm_test.png', confusion_matrix)
plt.show()
confusion_matrix = cm(train_x, train_y)
import string
plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.imsave('../Results2/cm_train.png', confusion_matrix)
plt.show()
confusion_matrix = cm(valid_x, valid_y)
import string
plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.imsave('../Results2/cm_val.png', confusion_matrix)
plt.show()

