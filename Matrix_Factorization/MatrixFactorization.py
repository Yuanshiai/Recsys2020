import numpy as np
import time
import math
import pandas as pd
import os


'''
参数说明：
    R:用户-物品的共现矩阵  m*n
    P:用户因子矩阵 m*d
    Q:物品因子矩阵 d*n
    d:隐向量的维度
    steps:最大迭代次数
    alpha:学习率
    Lambda:L2正则化的权重系数
'''


def matrix_factorization(R, P, Q, d, steps, alpha=0.05, Lambda=0.002):
    # 总时长
    sum_st = 0
    # 前一次的损失
    e_old = 0
    flag = 1
    # 梯度下降结束条件1：满足最大迭代次数
    for step in range(steps):
        st = time.time()
        # 本次的损失大小
        e_new = 0
        for u in range(len(R)):
            for i in range(len(R[u])):
                if R[u][i] > 0:
                    eui = R[u][i] - np.dot(P[u, :], Q[:, i])
                    for k in range(d):
                        temp = P[u][k]
                        P[u][k] = P[u][k] + alpha * eui * Q[k][i] - Lambda * P[u][k]
                        Q[k][i] = Q[k][i] + alpha * eui *temp - Lambda * Q[k][i]
        cnt = 0
        for u in range(len(R)):
            for i in range(len(R[u])):
                if R[u][i] > 0:
                    cnt += 1
                    e_new = e_new + pow(R[u][i]-np.dot(P[u, :], Q[:, i]), 2)
        et = time.time()
        e_new = e_new / cnt
        # 第一次不计算前后损失的差值
        if step == 0:
            e_old = e_new
            continue
        sum_st = sum_st + (et - st)
        # 梯度下降结束条件2：loss过小，结束迭代
        if e_new < 1e-3:
            flag = 2
            break
        # 梯度下降结束条件3：前后loss之差过小，梯度消失，跳出
        if (e_old - e_new) < 1e-10:
            flag = 3
            break
        else:
            e_old = e_new
    print(f'-------------Summary------------\nType of jump out:{flag}\nTotal steps:{step + 1}\n'
          f'Total time:{sum_st}\nAverage time:{sum_st / (step + 1)}\nThe e is:{e_new}')
    return P, Q


# 获取全部用户和项目
def getUser_Item(dsname, dformat):
    st = time.time()
    train = pd.read_csv(dsname+"_Train.txt", sep=',', header=None, names=dformat)
    test = pd.read_csv(dsname+"_test.txt", sep=',', header=None, names=dformat)
    data = pd.concat([train, test])
    all_user = np.unique(data['user'])
    all_item = np.unique(data['item'])
    train.sort_values(by=['user', 'item'], axis=0, inplace=True)
    num_user = max(all_user)+1
    num_item = max(all_item)+1
    rating = np.zeros([num_user, num_item], dtype=int)
    for i in range(len(train)):
        user = train.iloc[i]['user']
        item = train.iloc[i]['item']
        score = train.iloc[i]['rating']
        rating[user][item] = score
    if os.path.exists("D:\\YSA\\MovieLens\\ml-100k\\Basic_MF"):
        pass
    else:
        os.mkdir("D:\\YSA\\MovieLens\\ml-100k\\Basic_MF")
    np.savetxt("D:\\YSA\\MovieLens\\ml-100k\\Basic_MF\\rating.txt", rating, fmt="%d", delimiter=',', newline='\n')
    et = time.time()
    print(f'Total time:{et - st}')
    return rating


def topK(dic, k):
    keys = []
    values = []
    for i in range(0, k):
        key, value = max(dic.items(), key=lambda x: x[1])
        keys.append(key)
        values.append(value)
        dic.pop(key)
    return keys, values


def getData(dfomat):
    rating = np.loadtxt("D:\\YSA\\MovieLens\\ml-100k\\Basic_MF\\rating.txt", delimiter=',', dtype=float)
    train = pd.read_csv("D:\\YSA\\MovieLens\\ml-100k\\ML100K_Train.txt", sep=',', header=None, names=dfomat)
    test = pd.read_csv("D:\\YSA\\MovieLens\\ml-100k\\ML100K_test.txt", sep=',', header=None, names=dfomat)
    data = pd.concat([train, test])
    all_user = np.unique(data[str(dfomat[0])])
    all_item = np.unique(data[str(dfomat[1])])
    return rating, train, test, all_user, all_item


def train(rating, d, steps):
    R = rating
    M = len(R)
    N = len(R[0])
    # P,Q初始化
    P = np.random.normal(loc=0, scale=0.01, size=(M, d))
    Q = np.random.normal(loc=0, scale=0.01, size=(d, N))
    P, Q = matrix_factorization(R, P, Q, d, steps)
    if os.path.exists("D:\\YSA\\MovieLens\\ml-100k\\Basic_MF"):
        pass
    else:
        os.mkdir("D:\\YSA\\MovieLens\\ml-100k\\Basic_MF")
    np.savetxt("D:\\YSA\\MovieLens\\ml-100k\\Basic_MF\\userMatrix.txt", P, fmt="%.6f", delimiter=',', newline='\n')
    np.savetxt("D:\\YSA\\MovieLens\\ml-100k\\Basic_MF\\itemMatrix.txt", Q, fmt="%.6f", delimiter=',', newline='\n')


def test(trainData, testData, all_item, k):
    P = np.loadtxt("D:\\YSA\\MovieLens\\ml-100k\\Basic_MF\\userMatrix.txt", delimiter=',', dtype=float)
    Q = np.loadtxt("D:\\YSA\\MovieLens\\ml-100k\\Basic_MF\\itemMatrix.txt", delimiter=',', dtype=float)

    valid_cnt = 0
    Hits = 0
    MRR = 0
    NDCG = 0
    st = time.time()
    test_user = np.unique(testData['user'])
    for user in test_user:
        visited_item = list(trainData[trainData['user'] == user]['item'])
        if len(visited_item) == 0:
            continue
        # 去除测试集中同一用户的相同访问记录
        testlist = list(testData[testData['user'] == user]['item'].drop_duplicates())
        testlist = list(set(testlist) - set(testlist).intersection(set(visited_item)))
        if len(testlist) == 0:
            continue
        valid_cnt += 1
        poss = {}
        for item in all_item:
            if item in visited_item:
                continue
            else:
                poss[item] = np.dot(P[user, :], Q[:, item])
        rankedList, test_score = topK(poss, k)
        h = list(set(testlist).intersection(set(rankedList)))

        Hits += len(h)
        for item in testlist:
            for i in range(len(rankedList)):
                if rankedList[i] == item:
                    MRR += 1 / (i+1)
                    NDCG += 1 / (math.log2(i+2))
                else:
                    continue

    HR = Hits / len(testData)
    MRR /= len(testData)
    NDCG /= len(testData)
    print('HR@10:%.4f\nMRR@10:%.4f\nNDCG@10:%.4f' % (HR, MRR, NDCG))


    '''
    R = [[5, 2, 0, 3, 1],
         [0, 2, 1, 4, 5],
         [1, 1, 0, 2, 4],
         [2, 2, 0, 5, 0]]
    '''












if __name__ == '__main__':
    rtnames = ["user", "item", "rating", "time"]
    dsname = "D:\\YSA\\MovieLens\\ml-100k\\ML100K"
    rating, trainData, testData, all_user, all_item = getData(rtnames)
    # train(rating, 30, 10)
    test(trainData, testData, all_item, 10)




