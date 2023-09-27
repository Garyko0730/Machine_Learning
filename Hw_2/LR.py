# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 20:06:38 2023

@author: 21人智高鹏
"""

import numpy as np
import matplotlib.pyplot as plt

def cost_gradient(W, X, Y, n):
    # 计算梯度
    G = np.dot(X.T, (np.dot(X, W) - Y)) / n
    # 计算当前W对应的损失
    j = np.sum((np.dot(X, W) - Y) ** 2) / (2 * n)
    return (j, G)

def gradientDescent(W, X, Y, lr, iterations):
    n = np.size(Y)
    J = np.zeros(iterations)
    
    for i in range(iterations):
        (J[i], G) = cost_gradient(W, X, Y, n)
        # 更新W
        W = W - lr * G
    return (W, J)

iterations = 500  # 迭代次数
lr = 0.001/2 # 学习率

data = np.loadtxt('LR.txt', delimiter=',')

n = np.size(data[:, 1])
W = np.zeros((2, 1))
X = np.c_[np.ones(n), data[:, 0]]
Y = data[:, 1].reshape(n, 1)

(W, J) = gradientDescent(W, X, Y, lr, iterations)

# 绘制图形
plt.figure()
plt.plot(data[:, 0], data[:, 1], 'rx')
plt.plot(data[:, 0], np.dot(X, W))

plt.figure()
plt.plot(range(iterations), J)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.show()