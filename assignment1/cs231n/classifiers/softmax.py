from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]
    num_classes = W.shape[1]

    f = X.dot(W)

    for i in range(num_train):
        f_i = f[i,:]
        log_C_i = np.max(f_i)
        f_i += -log_C_i #for numeric stability
        loss_i = -f_i[y[i]]+np.log(np.sum(np.exp(f_i)))
        loss += loss_i

        for j in range(num_classes):
            softmax_prob = np.exp(f_i[j])/sum(np.exp(f_i))
            dW[:,j] += - ((y[i] == j)-softmax_prob)*X[i]

    loss = loss/num_train + reg*np.sum(W*W)
    dW = dW/num_train + reg*W


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]
    num_classes = W.shape[1]

    f = X.dot(W)
    log_C = np.max(f,1).reshape(num_train,-1)
    f -= log_C

    #calculating loss
    f_y_i = f[range(num_train),y].reshape(-1,1)
    temp = np.log(np.sum(np.exp(f),1)).reshape(-1,1)
    loss = np.sum(- f_y_i + temp)/num_train + 0.5*reg*np.sum(W*W)

    #calculating gradient

    softmax_prob = np.exp(f)/np.sum(np.exp(f),1).reshape(-1,1)
    softmax_prob[range(num_train),list(y)] += -1

    dW += X.T.dot(softmax_prob)/num_train + reg*W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
