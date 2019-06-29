import numpy as np
from matplotlib import pyplot as plt


N = 1000 # # of points per class
D = 2 # dimensionality
K = 3 # # of calss
X = np.zeros((N*K,D)) # data matrix
y = np.zeros(N*K, dtype = 'uint8') # class label
n = X.shape[0]

for j in range(K):
    idx = range(N*j,N*(j+1))
    r = np.linspace(0.0,1,N) #radius
    t = np.linspace(j*4,(j+1)*4, N) + np.random.randn(N)*.2 # theta
    X[idx] = np.c_[r*np.sin(t), r*np.cos(t)]
    y[idx] = j

h = 100 # size of hidden layer
W1 = 0.01 * np.random.randn(D,h)
b1 = np.zeros((1,h))
W2 = 0.01 * np.random.randn(h,K)
b2 = np.zeros((1,K))

#Setting some hyperparameters
step_size = .5e-1
reg = 1e-3


##fitting NN with gradient descent
for i in range(10001):

    # forward pass : evaluate class scores with a 2-layer Nerual Network
    hidden_layer = np.maximum(0, np.dot(X,W1)+b1) # ReLu
    scores = np.dot(hidden_layer,W2) + b2

    exp_scores = np.exp(scores)
    probs = exp_scores/np.sum(exp_scores, 1, keepdims=True )

    #compute the loss : avg cross entropy loss and L2 reg
    cor_probs = probs[range(n),y]
    data_loss = np.sum(-np.log(cor_probs))/n
    reg_loss = .5*reg*np.sum(W1*W1) + .5*reg*np.sum(W2*W2)
    loss = data_loss + reg_loss

    if i % 1000 == 0:
        pred = np.argmax(scores,1) ; pred.shape
        acc = np.mean(pred==y)
        print(f"iteration {i}: loss {loss}")
        print(f"    and Accuracy is {acc}")

    #Backprop
    dscores = probs
    dscores[range(n),y] -= 1
    dscores /= n

    #Backprop the gradient of parameters
    dW2 = np.dot(hidden_layer.T,dscores) #; print(dW2.shape)
    db2 = np.sum(dscores,0, keepdims = True) #; print(db2.shape)

    dhidden = np.dot(dscores, W2.T)#; print(dhidden.shape)
    dhidden[hidden_layer < 0] = 0

    dW1 = np.dot(X.T, dhidden)#; print(dW1.shape)
    db1 = np.sum(dhidden,0,keepdims=True)#; print(db1.shape)

    dW2 += reg * W2
    dW1 += reg * W1

    # perfom  parameter update
    W1 += -step_size * dW1
    b1 += -step_size * db1
    W2 += -step_size * dW2
    b2 += -step_size * db2
