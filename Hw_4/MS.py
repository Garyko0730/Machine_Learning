# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def read_data(addr):
    data = np.loadtxt(addr, delimiter=',')

    n = data.shape[0]
    
    ###### You may modify this section to change the model
    degree = 6 # 多项式的阶数
    # 生成多项式特征
    X_polynomial = np.ones([n, 1])  # 初始化多项式特征矩阵为全为1的列
    for d in range(1, degree + 1):
        X_polynomial = np.concatenate([X_polynomial, data[:, 0:6] ** d], axis=1)
    X = np.concatenate([np.ones([n, 1]), X_polynomial], axis=1)
    ###### You may modify this section to change the model
    
    Y = None
    if "train" in addr:
        Y = np.expand_dims(data[:, 6], axis=1)
    
    return (X,Y,n)

def cost_gradient(W, X, Y, n):
    G = X.T@((1/(1+np.exp(-(X@W))))-Y)###### Gradient
    j = np.mean(np.sum(-Y*np.log((1/(1+np.exp(-(X@W)))))-(1-Y)*np.log(1-(1/(1+np.exp(-(X@W)))))))###### cost with respect to current W
    
    return (j, G)

def train(W, X, Y, lr, n, iterations):
    k = 10  # 10-fold cross-validation
    fold_size = n // k
    J = np.zeros([iterations, 1])
    E_trn = np.zeros([iterations, 1])
    E_val = np.zeros([iterations, 1])

    for i in range(iterations):
        for j in range(k):
            start = j * fold_size
            end = (j + 1) * fold_size

            X_val = X[start:end]
            Y_val = Y[start:end]

            X_trn = np.concatenate([X[:start], X[end:]], axis=0)
            Y_trn = np.concatenate([Y[:start], Y[end:]], axis=0)

            (J[i], G) = cost_gradient(W, X_trn, Y_trn, X_trn.shape[0])
            W = W - lr * G

            E_trn[i] += error(W, X_trn, Y_trn)
            E_val[i] += error(W, X_val, Y_val)

        E_trn[i] /= k
        E_val[i] /= k

    print(E_val[-1])

    return (W, J, E_trn, E_val)

def error(W, X, Y):
    Y_hat = 1 / (1 + np.exp(-X@W))
    Y_hat[Y_hat<0.5] = 0
    Y_hat[Y_hat>0.5] = 1
    
    return (1-np.mean(np.equal(Y_hat, Y)))

def predict(W):
    (X, _, _) = read_data("test_data.csv")
    
    Y_hat = 1 / (1 + np.exp(-X@W))
    Y_hat[Y_hat<0.5] = 0
    Y_hat[Y_hat>0.5] = 1
    
    idx = np.expand_dims(np.arange(1,201), axis=1)
    np.savetxt("predict.csv", np.concatenate([idx, Y_hat], axis=1), header = "Index,ID", comments='', delimiter=',')

iterations = 5000###### Training loops
lr = 0.00001###### Learning rate

(X, Y, n) = read_data("train.csv")
W = np.random.random([X.shape[1], 1])

(W,J,E_trn,E_val) = train(W, X, Y, lr, n, iterations)

###### You may modify this section to do 10-fold validation
# 绘制成本曲线
plt.figure()
plt.plot(range(iterations), J)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost during Training')
plt.show()

plt.figure()
plt.ylim(0, 1)
plt.plot(range(iterations), np.mean(E_trn, axis=1), "b", label='Training Error')
plt.plot(range(iterations), np.mean(E_val, axis=1), "r", label='Validation Error')
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.title('Error during Training')
plt.legend()
plt.show()
###### You may modify this section to do 10-fold validation

predict(W)
