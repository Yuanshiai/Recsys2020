import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import math


# sigmoid函数将任意实数值映射至0~1范围内
def sigmoid(x):
    if x >= 0:
        return 1 / (1 + np.exp(-x))
    else:
        return np.exp(x) / (1 + np.exp(x))


# 交叉熵损失函数
def cross_entropy(y, x):
    if x >= 0:
        return (1-y) * x + np.log(1 + np.exp(-x))
    else:
        return (-x) * y + np.log(1 + np.exp(x))


def softmax(X, k):
    if k >= len(X):
        return -1
    else:
        X = np.array(X)
        # 获得X中的最大值
        val_max = max(X)
        sum_X = 0
        for i in range(len(X)):
            sum_X += np.exp(X[i] - val_max)
        return np.exp(X[k] - val_max) / sum_X


def logistic_regression(X, Y, steps, alpha=0.2):
    m, n = np.shape(X)
    num_feat = n
    # 初始化w、b参数
    w = np.random.normal(loc=0, scale=0.1, size=num_feat)
    b = 0.0
    y_hat = np.zeros(m)
    # 梯度下降停止条件1:满足迭代次数
    for step in range(steps):
        for i in range(m):
            z = np.dot(X[i], w.transpose()) + b
            y_hat[i] = sigmoid(z)
        error = y_hat - Y
        w = w - alpha * np.dot(error, X)
        b = b - alpha * np.sum(error)
    return w, b


def predict(X, w, b):
    y_pre = []
    for i in range(np.shape(X)[0]):
        z = np.dot(X[i], w.T) + b
        y_hat = sigmoid(z)
        if y_hat >= 0.5:
            y_pre.append(1)
        else:
            y_pre.append(0)
    return y_pre


if __name__ == '__main__':
    featName = ['Number', 'Plasma', 'Diastolic', 'Triceps', '2-Hour', 'Body', 'Diabetes', 'Age', 'Class']
    path = "D:\\YSA\\dataFile\\GBDT_LR\\"
    X_train = np.loadtxt(path+"train.txt", delimiter=',', dtype=int)
    w, b = logistic_regression(X_train[:, :-1], X_train[:, -1], steps=1000, alpha=0.01)
    X_test = np.loadtxt(path+"test.txt", delimiter=',', dtype=int)
    y_pre = predict(X_test[:, :-1], w, b)
    # print(y_pre)
    acc = accuracy_score(y_pre, X_test[:, -1])
    print(acc)




