import numpy as np
import matplotlib.pyplot as plt

#Generating toy data
N = 100 # # of points per class
D = 2 # dimensionality
K = 3 # # of calss
X = np.zeros((N*K,D)) # data matrix
y = np.zeros(N*K, dtype = 'uint8') # class label

for j in range(K):
    idx = range(N*j,N*(j+1))
    r = np.linspace(0.0,1,N) #radius
    t = np.linspace(j*4,(j+1)*4, N) + np.random.randn(N)*.2 # theta
    X[idx] = np.c_[r*np.sin(t), r*np.cos(t)]
    y[idx] = j

# draw plot
plt.scatter(X[:,0],X[:,1], c=y, s=20, cmap=plt.cm.Spectral)
plt.show()

#########################
## softx max clasifier ##
#########################

# initialize parameters randomly
W = 0.01 * np.random.randn(D,K)
b = np.zeros((1,K))

# Setting hyperparameters
step_size = .5e-1
reg = 1e-3

# gradient descent loop
n = X.shape[0] ; print(n)

for i in range(300):

    #evaluate class scores, [N x K]
    scores = np.dot(X,W) + b

    #compute the class prob
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, 1, keepdims=True) # can using reshape(-1,1) instead keepdims opt

    #compute the loss : avg cross-entropy loss with L2 reg
    correct_prob = probs[range(n),y]
    data_loss = np.sum(-np.log(correct_prob))/n
    reg_loss = .5*reg*np.sum(W*W)
    loss = data_loss + reg_loss

    if i % 30 == 0 :
        pred = np.argmax(probs, 1)
        acc = np.mean(pred ==y)
        print(f"Iteration {i} : loss {loss}")
        print(f"    And Accuracy is {acc}")
    #compute the gradient on scores
    dscores = probs
    dscores[range(n),y] -= 1
    dscores /= n

    #compute the gradients on each params
    dW = X.T.dot(dscores)
    db = np.sum(dscores, 0, keepdims=True)
    dW += reg*W

    #perfom ap parameter update
    W += -step_size*dW
    b += -step_size*db
