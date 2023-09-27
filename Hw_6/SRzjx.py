import numpy as np
import matplotlib.pyplot as plt
    
def cost_gradient(W, X, Y, n):
    z = X @ W  
    Y_hat = np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
    G = X.T @ (Y_hat - Y)
    j = (-1 / n) * np.sum(Y * np.log(Y_hat))

    return (j, G)

def train(W, X, Y, n, lr, iterations):
    J = np.zeros([iterations, 1])

    for i in range(iterations):
        (J[i], G) = cost_gradient(W, X, Y, n)
        W = W - lr * G

    return (W, J)

def error(W, X, Y):
    z = X @ W
    Y_hat = np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
    pred = np.argmax(Y_hat, axis=1)
    label = np.argmax(Y, axis=1)
    
    return 1 - np.mean(pred == label)

iterations = 1000
lr = 0.0001

data = np.loadtxt('SR.txt', delimiter=',')

n = data.shape[0]
X = np.concatenate([np.ones([n, 1]), data[:, :3]], axis=1)
Y = data[:, 3].astype(np.int32)
c = np.max(Y) + 1
Y = np.eye(c)[Y]

W = np.random.random([X.shape[1], c])

(W, J) = train(W, X, Y, n, lr, iterations)

plt.figure()
plt.plot(range(iterations), J)

print(error(W, X, Y))