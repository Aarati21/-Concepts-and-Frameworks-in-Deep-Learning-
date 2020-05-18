import numpy as np
from util import *
# do not include any more libraries here!
# do not put any code outside of functions!

# Q 2.1
# initialize b to 0 vector
# b should be a 1D array, not a 2D array with a singleton dimension
# we will do XW + b. 
# X be [Examples, Dimensions]
def initialize_weights(in_size,out_size,params,name=''):
    W, b = None, None
    b = np.zeros(out_size)
    W = np.random.uniform(low = -np.sqrt(6.0/(in_size+out_size)), high = np.sqrt(6.0/(in_size+out_size)), size = (in_size, out_size))
    
    params['W' + name] = W
    params['b' + name] = b

# Q 2.2.1
# x is a matrix
# a sigmoid activation function
def sigmoid(x):
    res = None
    res = 1.0/(1.0+np.exp(-x))
    return res

# Q 2.2.2
def forward(X,params,name='',activation=sigmoid):
    """
    Do a forward pass

    Keyword arguments:
    X -- input vector [Examples x D]
    params -- a dictionary containing parameters
    name -- name of the layer
    activation -- the activation function (default is sigmoid)
    """
    pre_act, post_act = None, None
    # get the layer parameters
    W = params['W' + name]
    b = params['b' + name]

    # your code here
    pre_act = X.dot(W) + b
    post_act = activation(pre_act)

    # store the pre-activation and post-activation values
    # these will be important in backprop
    params['cache_' + name] = (X, pre_act, post_act)

    return post_act

# Q 2.2.2 
# x is [examples,classes]
# softmax should be done for each row
def softmax(x):
    
    res = None
    val = x-np.max(x, axis=1).reshape(-1,1)
    num = np.exp(val)
    denom = np.sum(num, axis = 1).reshape(-1,1)
    res = num/denom
    return res

# Q 2.2.3
# compute total loss and accuracy
# y is size [examples,classes]
# probs is size [examples,classes]
def compute_loss_and_acc(y, probs):
    
    loss, acc = None, None
    loss = -np.sum(np.multiply(y, np.log(probs)))
    y_index = np.argmax(y, axis = -1)
    
    probs_index = np.argmax(probs, axis = -1)
    
    acc = np.sum(np.equal(y_index, probs_index))/float(y.shape[0])
    return loss, acc 


# we give this to you
# because you proved it
# it's a function of post_act
def sigmoid_deriv(post_act):
    res = post_act*(1.0-post_act)
    return res

def backwards(delta,params,name='',activation_deriv=sigmoid_deriv):
    """
    Do a backwards pass

    Keyword arguments:
    delta -- errors to backprop
    params -- a dictionary containing parameters
    name -- name of the layer
    activation_deriv -- the derivative of the activation_func
    """
    grad_X, grad_W, grad_b = None, None, None
    # everything you may need for this layer
    W = params['W' + name]
    b = params['b' + name]
    X, pre_act, post_act = params['cache_' + name]
    # your code here
    # do the derivative through activation first
    # then compute the derivative W,b, and X
    
    act_post = activation_deriv(post_act)
    grad_W = (X.T).dot(delta*act_post)
    grad_X = (delta*act_post).dot(W.T)
    m = np.ones((1, delta.shape[0]))
    grad_b = m.dot(delta*act_post)
    grad_b = grad_b.reshape(-1)
    # store the gradients
    params['grad_W' + name] = grad_W
    params['grad_b' + name] = grad_b
    return grad_X

# Q 2.4
# split x and y into random batches
# return a list of [(batch1_x,batch1_y)...]
def get_random_batches(x,y,batch_size):
    batches = []
    
    n_batches = int(x.shape[0]/batch_size)
    for i in range(n_batches):
        random_index = np.random.choice(range(x.shape[0]), size = batch_size, replace = False)
        batches.append((x[random_index], y[random_index]))
    return batches
