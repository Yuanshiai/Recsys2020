"""
Factorization Machines(因子分解机)模型算法：稀疏矩阵下的二阶特征组合问(个性化推荐)
1、应用矩阵分解思想，引入隐向量构造FM模型方程
2、目标函数(损失函数复合FM模型方程)的最优问题：链式求导
3、SGD优化目标函数
"""
import numpy as np
import time
import pandas as pd
from random import normalvariate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 二分类输出非线性映射
def sigmoid(x):
    if x >= 0:
        return 1 / (1 + np.exp(-x))
    else:
        return np.exp(x) / (1 + np.exp(x))


# 计算logit损失函数:
def logit(y, y_hat):
    z = -y * y_hat
    if z <= 128:
        return np.log(1 + np.exp(z))
    else:
        return z


# 计算logit损失函数的外层偏导数（不含y_hat自身的一阶导数）
def df_logit(y, y_hat):
    return sigmoid(-y * y_hat) * (-y)


# FM的模型方程:LR线性组合+特征交叉项组合 = 一阶线性组合 + 二阶线性组合
def FM(X_i, w_0, W, V):
    # 样本X_i的特征分量xi和xj的2阶交叉项组合系数wij  = xi和xj对应的隐向量Vi和Vj的内积
    # 向量形式：Wij=<Vi,Vj> * xi * xj
    sum_interation = np.sum((X_i.dot(V)) ** 2 - (X_i ** 2).dot(V ** 2)) / 2
    # 预测函数值
    y_hat = w_0 + X_i.dot(W) + sum_interation
    return y_hat[0]


# SGD更新FM模型的参数列表: [w_0, W, V]
def FM_SGD(X, y, k=2, alpha=0.01, iter=50):
    m, n = np.shape(X)
    # 参数w_0,W初始化: w_0=0, W=(n, 1)
    w_0, W = 0, np.zeros((n, 1))
    # 参数V初始化: V=(n, k)~N(0,1)
    V = np.random.normal(loc=0, scale=1, size=(n, k))
    # FM模型的参数列表:[w_0, W, V]
    all_FM_params = []
    # 前一次的总损失
    loss_total_old = 0
    # SGD开始的时间
    st = time.time()
    # SGD结束条件1：满足最大迭代次数
    for step in range(iter):
        # SGD结束的标识
        flag = 1
        # 本次的总损失
        loss_total_new = 0
        # 遍历整个训练集
        for i in range(m):
            y_hat = FM(X_i=X[i], w_0=w_0, W=W, V=V)
            loss_total_new += logit(y=y[i], y_hat=y_hat)
            # logit损失函数的外层偏导数
            df_loss = df_logit(y[i], y_hat)
            # 计算logit损失函数对w0的偏导数
            df_w0_loss = df_loss
            # 更新参数w_0
            w_0 = w_0 - alpha * df_w0_loss
            # 遍历n维向量X[i]
            for j in range(n):
                if X[i, j] == 0:
                    continue
                # 计算logit损失函数对Wij的偏导数
                df_Wij_loss = df_loss * X[i, j]
                # 更新参数Wij
                W[j] = W[j] - alpha * df_Wij_loss
                # 遍历k维隐向量Vj
                for f in range(k):
                    # 计算logit损失函数对Vjf的偏导数
                    df_Vjf_loss = df_loss * X[i, j] * (X[i].dot(V[:, f]) - X[i, j] * V[j, f])
                    # 更新参数Vjf
                    V[j, f] = V[j, f] - alpha * df_Vjf_loss
        # SGD结束条件2：总损失过小，跳出
        if loss_total_new < 1e-2:
            flag = 2
            break
        # 如果是第一次计算总损失，则不计算前后损失之差
        if step == 0:
            loss_total_old = loss_total_new
            continue
        # SGD结束条件3：前后损失变化过小，跳出
        if (loss_total_old - loss_total_new) < 1e-5:
            flag = 3
            break
        else:
            loss_total_old = loss_total_new
        # 每迭代10步输出一次loss
        if step % 10 == 0:
            print(f'the step is:{step+1},the loss is:{loss_total_new}')
        all_FM_params.append([w_0, W, V])
    # SGD结束时间
    et = time.time()
    print("the total time is:%.4f\nthe type of jump out:%d\nthe total step:%d\nthe loss is:%.6f" % ((et - st), flag, step+1, loss_total_old))

    return all_FM_params


# FM模型预测测试集分类结果
def FM_predic(X, w_0, W, V):
    # sigmoid阙值设置
    predicts, threshold = [], 0.5
    # 遍历测试集
    for i in range(X.shape[0]):
        # X[i]的预测值
        y_hat = FM(X[i], w_0, W, V)
        # 分类结果非线性映射
        predicts.append(-1 if sigmoid(y_hat) < threshold else 1)
    return np.array(predicts)


# 计算准确度得分
def accuracy_score(Y, predicts):
    # 统计预测正确的个数
    hits_count = 0
    # 准确度
    score_acc = 0
    for i in range(Y.shape[0]):
        if Y[i] == predicts[i]:
            hits_count += 1
    score_acc = hits_count / Y.shape[0]
    return score_acc


# FM在不同迭代次数下的参数列表，训练集的损失值和测试集的准确率变化
def draw_research(all_FM_params, X_train, y_train, X_test, y_test):
    loss_total_all, accuracy_total_all = [], []
    for w_0, W, V in all_FM_params:
        loss_total = 0
        for i in range(X_train.shape[0]):
            loss_total += logit(y=y_train[i], y_hat=FM(X_train[i], w_0, W, V))
        loss_total_all.append(loss_total / X_train.shape[0])
        accuracy_total_all.append(accuracy_score(y_test, FM_predic(X_test, w_0, W, V)))
    plt.plot(np.arange(len(all_FM_params)), loss_total_all, color='#FF4040', label='训练集的损失值')
    plt.plot(np.arange(len(all_FM_params)), accuracy_total_all, color='#4876FF', label='测试集的准确率')
    plt.xlabel('SGD迭代次数')
    plt.title('FM模型:二阶互异特征组合')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    np.random.seed(123)
    df = pd.read_csv("D:\\YSA\\dataFile\\xg.csv", sep=',')
    # 标签列从[0, 1]离散到[-1, 1]
    df['Class'] = df['Class'].map({0: -1, 1: 1})
    # 分割数据集，生成训练集，测试集
    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1].values, df.iloc[:, -1].values, test_size=0.3, random_state=123)
    # 归一化训练集，返回[0, 1]区间
    X_train = MinMaxScaler().fit_transform(X_train)
    # 归一化测试集，返回[0, 1]区间
    X_test = MinMaxScaler().fit_transform(X_test)
    # FM模型预测
    all_FM_params = FM_SGD(X_train, y_train, k=2, alpha=0.01, iter=45)
    # FM模型的参数列表
    w_0, W, V = all_FM_params[-1]
    # FM模型预测测试集分类结果
    predicts = FM_predic(X_test, w_0, W, V)
    # 准确率
    acc = accuracy_score(y_test, predicts)
    print(f'FM模型在测试集分类的准确率:{acc}')
    #draw_research(all_FM_params, X_train, y_train, X_test, y_test)




