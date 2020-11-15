import numpy as np
import copy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn import tree
import sys
import os
os.environ['PATH'] = os.pathsep + 'C:\\Program Files\\Graphviz 2.44.1\\bin'

class Node:
    """
    树的节点类
    """

    def __init__(self, feature=-1, split_val=None, results=None, left=None, right=None):
        """
        :param feature: 用于切分数据集的特征索引
        :param split_val: 设置切分的值
        :param results: 存储节点的值
        :param left: 左子树
        :param right: 右子树
        """

        self.feature = feature
        self.split_val = split_val
        self.results = results
        self.left = left
        self.right = right



def leaf(dataSet):
    """计算节点的数值
    :param dataSet: {ndarray}训练样本
    :return: 均值
    """

    len = np.shape(np.unique(dataSet[:, -1]))[0]
    if len == 1:
        return int(dataSet[0, -1])
    elif np.count_nonzero(dataSet[:, -1]) > np.shape(dataSet)[0] / 2:
        return 1
    else:
        return 0


def Gini(dataSet):
    m = np.shape(dataSet)[0]
    num_nonzero = np.count_nonzero(dataSet[:, -1])
    p = num_nonzero / m
    return 2 * p * (1 - p)


def err_cnt(dataSet):
    """计算误差
    :param dataSet: {ndarray}训练数据
    :return: 总方差
    """

    return np.var(dataSet[:, -1]) * np.shape(dataSet)[0]


def split_tree(dataSet, feature, split_val):
    """根据特征feature中的值split_val将数据集data划分为左右子树
    :param data: {list}训练样本
    :param feature: {int}需要划分的特征索引
    :param split_val: {float}指定的划分值
    :return:(set_1, set_2): {tuple} 左右子树的集合
    """

    set_L = dataSet[np.nonzero(dataSet[:, feature] <= split_val)[0], :]
    set_R = dataSet[np.nonzero(dataSet[:, feature] > split_val)[0], :]
    return set_L, set_R


class CART_regression(object):
    """
    CART算法类
    """

    def __init__(self, X, Y, min_sample, min_err, max_height=20):
        """
        :param X: 回归样本数据的特征
        :param Y: 回归样本数据的标签
        :param min_sample: 每个叶节点最少样本数
        :param min_err: 最小损失
        """
        self.X = X
        self.Y = Y
        self.min_sample = min_sample
        self.min_err = min_err
        self.max_height = max_height

    def fit(self):
        """
        构建树
        input:data{list} -- 训练样本
              min_sample{int} -- 叶子节点中最少样本数
              min_err{float} -- 最小的error
        output: node:树的根节点
        """
        # 将样本特征与样本标签合成完整的样本
        data = np.c_[self.X, self.Y]
        # 初始化
        best_err = Gini(data)
        # 存储最佳切分属性及最佳切分点
        bestCriteria = None
        # 存储切分后的两个数据集
        bestSets = None
        # 构建决策树，返回该决策树的根节点
        '''
        生成叶子节点的3个条件：
        1.数据集样本数小于给定的最小样本数
        2.CART树达到给定的最大深度
        3.数据集样本的gini指数小于给定的最小误差
        '''
        if np.shape(data)[0] <= self.min_sample or self.max_height == 1 or best_err <= self.min_err :
            return Node(results=leaf(data))

        # 开始构建CART回归树
        num_feature = np.shape(data[0])[0] - 1
        for feat in range(num_feature):
            val_feat = np.unique(data[:, feat])
            for val in val_feat:
                # 尝试划分
                set_L, set_R = split_tree(data, feat, val)
                if np.shape(set_L)[0] < 2 or np.shape(set_R)[0] < 2:
                    continue
                # 计算划分后的error值
                p = np.shape(set_L)[0] / np.shape(data)[0]
                err_now = p * Gini(set_L) + (1 - p) * Gini(set_R)
                # 更新最新划分
                if err_now < best_err:
                    best_err = err_now
                    bestCriteria = (feat, val)
                    bestSets = (set_L, set_R)
        # 生成左右子树
        left = CART_regression(bestSets[0][:, :-1], bestSets[0][:, -1], self.min_sample, self.min_err, self.max_height-1).fit()
        right = CART_regression(bestSets[1][:, :-1], bestSets[1][:, -1], self.min_sample, self.min_err, self.max_height-1).fit()
        return Node(feature=bestCriteria[0], split_val=bestCriteria[1], left=left, right=right)



def predict(sample, tree):
    f"""对每一个样本sample进行预测
    :param sample: {list}:样本
    :param tree: 训练好的CART回归模型
    :return: results{float} :预测值
    """

    # 叶子节点
    if tree.results is not None:
        return tree.results
    else:
        # 不是叶节点
        val_sample = sample[tree.feature]
        branch = None
        # 选择右子树
        if val_sample > tree.split_val:
            branch = tree.right
        else:
            branch = tree.left
        return predict(sample, branch)


def test(X, tree):
    """评估CART回归模型
    :param X: {list} 测试样本
    :param Y: {list} 测试标签
    :param tree: 训练好的CART回归树模型
    :return:  均方误差
    """

    m = np.shape(X)[0]
    y_hat = []
    for i in range(m):
        pre = predict(X[i], tree)
        y_hat.append(pre)
    return y_hat


def numLeaf(tree):
    if tree.results is not None:
        return 1
    else:
        return numLeaf(tree.left) + numLeaf(tree.right)


def heightTree(tree):
    if tree.results is not None:
        return 1
    else:
        heightL = heightTree(tree.left)
        heughtR = heightTree(tree.right)
        if heightL > heughtR:
            return heightL + 1
        else:
            return heughtR + 1
def showTree(tree):
    node = {}

    if tree.results is None:
        node['feat'] = tree.feature
        node['splitVal'] = tree.split_val
        print(node)
        showTree(tree.left)
        showTree(tree.right)
    else:
        node['value'] = tree.results
        print(node)


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
    cartTree = CART_regression(X_train, y_train, 10, 0.2, 8).fit()
    print(numLeaf(cartTree))
    print(heightTree(cartTree))
    showTree(cartTree)
    y_hat = test(X_test, cartTree)
    print(accuracy_score(y_test, y_hat))