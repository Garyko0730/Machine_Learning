# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def cost_gradient(W, X, Y, n):
    G =  X.T @ ( np.exp(X @ W) / np.sum(np.exp(X @ W), axis=1, keepdims=True) - Y)
    j = -np.mean(np.sum(Y * np.log( np.exp(X @ W) / np.sum(np.exp(X @ W), axis=1, keepdims=True)), axis=1))

    return (j, G)

def train(W, X, Y, n, lr, iterations):
    J = np.zeros([iterations, 1])

    for i in range(iterations):
        (J[i], G) = cost_gradient(W, X, Y, n)
        W = W - lr * G

    return (W, J)

def error(W, X, Y):
    Y_hat = np.exp(X @ W) / np.sum(np.exp(X @ W), axis=1, keepdims=True)
    pred = np.argmax(Y_hat, axis=1)
    label = np.argmax(Y, axis=1)

    return (1 - np.mean(np.equal(pred, label)))

iterations = 1500  # Training loops
lr = 0.0001  # Learning rate

data = np.loadtxt('SR.txt', delimiter=',')

n = data.shape[0]
X = np.concatenate([np.ones([n, 1]),
                    np.expand_dims(data[:, 0], axis=1),
                    np.expand_dims(data[:, 1], axis=1),
                    np.expand_dims(data[:, 2], axis=1)],
                   axis=1)
Y = data[:, 3].astype(np.int32)
c = np.max(Y) + 1
Y = np.eye(c)[Y]

W = np.random.random([X.shape[1], c])

(W, J) = train(W, X, Y, n, lr, iterations)

plt.figure()
plt.plot(range(iterations), J)
plt.xlabel('Iterations')  # 添加横轴标题
plt.ylabel('Cost')  # 添加纵轴标题
plt.title('Loss Curve')  # 添加图标题
print(error(W, X, Y))