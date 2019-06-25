import numpy as np

input_size = 4
hidden_size = 10
num_classes = 3
num_inputs = 5
std=1e-4

np.random.seed(0)
X = 10*np.random.randn(num_inputs , input_size)
y = np.array([0, 1, 2, 2, 1])

params = {}
params['W1'] = std * np.random.randn(input_size, hidden_size)
params['b1'] = np.zeros(hidden_size)
output_size = num_classes
params['W2'] = std * np.random.randn(hidden_size, output_size)
params['b2'] = np.zeros(output_size)


params['W1'].shape

W1, b1 = params['W1'], params['b1']
W2, b2 = params['W2'], params['b2']
N, D = X.shape
W2.shape
b2.shape

Relu = lambda x : x *(x>0)

H1 =  Relu(X.dot(W1)+b1)
X.shape
W1.shape
b1.shape

H1.shape

H2 = H1.dot(W2)+b2
H2.shape
scores = H2

loss_temp = np.exp(scores)/np.sum(1+np.exp(scores),0)
np.sum(loss_temp)

scores2 = scores - np.max(scores,0)
np.sum(1+np.exp(scores),0).shape

scores2 + 0.5reg
