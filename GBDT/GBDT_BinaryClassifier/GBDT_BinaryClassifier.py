import CART_regression_tree
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn import tree
import sys
import os
import numpy as np


def load_data(data_file):
    """导入训练数据
    :param data_file: {string} 保存训练数据的文件
    :return: {list} 训练数据
    """
    X, Y = [], []
    f = open(data_file)
    for line in f.readlines():
        sample = []
        lines = line.strip().split('\t')
        Y.append(lines[-1])
        for i in range(len(lines) - 1):
            sample.append(float(lines[i]))
        X.append(sample)
    return X, Y


# 二分类输出非线性映射
def sigmoid(x):
    if x >= 0:
        return 1 / (1 + np.exp(-x))
    else:
        return np.exp(x) / (1 + np.exp(x))


class GBDT_RT(object):
    """
    GBDT回归算法类
    """
    def __init__(self):
        self.trees = None
        self.learn_rate = None
        self.init_val = None
        self.leafNode = 0

    def get_init_value(self, y):
        """计算初始值的平均值
        :param y: {ndarray} 样本标签列表
        :return: average:{float} 样本标签的平均值
        """
        """
        初始化：
        F0(x)=log(p1/(1 - p1))
        """
        p = np.count_nonzero(y)
        n = np.shape(y)[0]
        return np.log(p / (n-p))

    def get_residuals(self, y, y_hat):
        """
        计算样本标签域预测列表的残差
        :param y: {ndarray} 样本标签列表
        :param y_hat: {ndarray} 预测标签列表
        :return: y_residuals {list} 样本标签与预测标签列表的残差
        """

        y_residuals = []
        for i in range(len(y)):
            y_residuals.append(y[i] - y_hat[i])
        return y_residuals

    def fit(self, X, Y, n_estimates, learn_rate, min_sample, min_err, max_height):
        """
        训练GDBT模型
        :param X: {list} 样本特征
        :param Y: {list} 样本标签
        :param n_estimates: {int} GBDT中CART树的个数
        :param learn_rate: {float} 学习率
        :param min_sample: {int} 学习CART时叶节点最小样本数
        :param min_err: {float} 学习CART时最小方差
        """

        # 初始化预测标签和残差
        self.init_val = self.get_init_value(Y)

        n = np.shape(Y)[0]
        F = np.array([self.init_val] * n)
        y_hat = np.array([sigmoid(self.init_val)] * n)
        y_residuals = Y - y_hat
        y_residuals = np.c_[Y, y_residuals]

        self.trees = []
        self.learn_rate = learn_rate
        # 迭代训练GBDT
        for j in range(n_estimates):
            tree = CART_regression_tree.CART_regression(X, y_residuals, min_sample, min_err, max_height).fit()
            self.leafNode += CART_regression_tree.numLeaf(tree)
            for k in range(n):
                res_hat = CART_regression_tree.predict(X[k], tree)[0]
                # 计算此时的预测值等于原预测值加残差预测值
                F[k] += self.learn_rate * res_hat
                y_hat[k] = sigmoid(F[k])
            y_residuals = Y - y_hat
            y_residuals = np.c_[Y, y_residuals]
            self.trees.append(tree)

    # 获取GBDT叶子节点个数
    def getGBDT_leafNode(self):
        return self.leafNode

    def GBDT_predicts(self, X_test):
        """
        预测多个样本
        :param X_test: {list} 测试集
        :return: predicts {list} 预测的结果
        """
        predicts = []
        for i in range(np.shape(X_test)[0]):
            pre_y = self.init_val
            for tree in self.trees:
                pre_y += self.learn_rate * CART_regression_tree.predict(X_test[i], tree)[0]
            if sigmoid(pre_y) >= 0.5:
                predicts.append(1)
            else:
                predicts.append(0)
        return predicts

    # GBDT叶子节点one-hot编码
    def GBDT_onehot(self, X):
        m = np.shape(X)[0]
        onehot = np.zeros((m, self.leafNode))
        for i in range(m):
            for tree in self.trees:
                leafIndex = CART_regression_tree.predict(X[i], tree)[1]
                onehot[i, leafIndex] = 1
        return onehot


    def cal_error(self, Y_test, predicts):
        """
        计算预测误差
        :param Y_test: {测试样本标签列表}
        :param predicts: {list} 测试样本预测列表
        :return: error {float} 均方误差
        """

        y_test = np.array(Y_test)
        y_predicts = np.array(predicts)
        error = np.square(y_test - y_predicts).sum() / len(Y_test)
        return error


if __name__ == '__main__':
    # name of features
    featName = ['Number', 'Plasma', 'Diastolic', 'Triceps', '2-Hour', 'Body', 'Diabetes', 'Age', 'Class']
    path = "D:\\YSA\\dataFile\\xg.csv"
    # read data file
    data = pd.read_csv(path, sep=',', header=0, names=featName)
    # set random seed
    np.random.seed(123)
    # split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1].values, data.iloc[:, -1].values,
                                                        test_size=0.2, random_state=123)
    # normalize train
    # X_train = MinMaxScaler().fit_transform(X_train)
    # normalize test
    # X_test = MinMaxScaler().fit_transform(X_test)
    gbdtTrees = GBDT_RT()
    gbdtTrees.fit(X_train, y_train, n_estimates=6, learn_rate=0.2, min_sample=30, min_err=0.3, max_height=4)
    for i in range(6):
        print(CART_regression_tree.numLeaf(gbdtTrees.trees[i]))
        print(CART_regression_tree.heightTree(gbdtTrees.trees[i]))
        CART_regression_tree.showTree(gbdtTrees.trees[i])
        print('--------------------------------------------')
    trainOnehot = gbdtTrees.GBDT_onehot(X_train)
    train = np.c_[trainOnehot, y_train]
    testOnehot = gbdtTrees.GBDT_onehot(X_test)
    test = np.c_[testOnehot, y_test]
    if os.path.exists("D:\\YSA\\dataFile\\GBDT_LR"):
        pass
    else:
        os.mkdir("D:\\YSA\\dataFile\\GBDT_LR")
    np.savetxt("D:\\YSA\\dataFile\\GBDT_LR\\train.txt", train, fmt="%d", delimiter=',', newline='\n')
    np.savetxt("D:\\YSA\\dataFile\\GBDT_LR\\test.txt", test, fmt="%d", delimiter=',', newline='\n')
    print("save successful!")
