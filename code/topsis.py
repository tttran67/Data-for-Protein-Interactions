import numpy as np
import pandas as pd
import math
import heapq
import csv


class AHP:
    """
    相关信息的传入和准备
    """

    def __init__(self, array):
        ## 记录矩阵相关信息
        self.array = array
        ## 记录矩阵大小
        self.n = array.shape[0]
        # 初始化RI值，用于一致性检验
        self.RI_list = [0, 0, 0.52, 0.89, 1.12, 1.26, 1.36, 1.41, 1.46, 1.49, 1.52, 1.54, 1.56, 1.58,
                        1.59]
        # 矩阵的特征值和特征向量
        self.eig_val, self.eig_vector = np.linalg.eig(self.array)
        # 矩阵的最大特征值
        self.max_eig_val = np.max(self.eig_val)
        # 矩阵最大特征值对应的特征向量
        self.max_eig_vector = self.eig_vector[:, np.argmax(self.eig_val)].real
        # 矩阵的一致性指标CI
        self.CI_val = (self.max_eig_val - self.n) / (self.n - 1)
        # 矩阵的一致性比例CR
        self.CR_val = self.CI_val / (self.RI_list[self.n - 1])

    """
    一致性判断
    """

    def test_consist(self):
        # 打印矩阵的一致性指标CI和一致性比例CR
        print("判断矩阵的CI值为：" + str(self.CI_val))
        print("判断矩阵的CR值为：" + str(self.CR_val))
        # 进行一致性检验判断
        if self.n == 2:  # 当只有两个子因素的情况
            print("仅包含两个子因素，不存在一致性问题")
        else:
            if self.CR_val < 0.1:  # CR值小于0.1，可以通过一致性检验
                print("判断矩阵的CR值为" + str(self.CR_val) + ",通过一致性检验")
                return True
            else:  # CR值大于0.1, 一致性检验不通过
                print("判断矩阵的CR值为" + str(self.CR_val) + "未通过一致性检验")
                return False

    """
    特征值法求权重
    """

    def cal_weight__by_eigenvalue_method(self):
        # 将矩阵最大特征值对应的特征向量进行归一化处理就得到了权重
        array_weight = self.max_eig_vector / np.sum(self.max_eig_vector)
        # 打印权重向量
        #print("特征值法计算得到的权重向量为：\n", array_weight)
        # 返回权重向量的值
        return array_weight

def entropy(x_mat):
    # 返回每个样本的指数
    # 样本数，指标个数
    n, m = np.shape(x_mat)
    # 一行一个样本，一列一个指标
    # 下面是标准化
    tep_x1 = (x_mat * x_mat).sum(axis=0)  # 每个元素平方后按列相加
    tep_x2 = np.tile(tep_x1, (n, 1))  # 将矩阵tep_x1平铺n行
    Z = x_mat / ((tep_x2) ** 0.5)  # Z为标准化矩阵

    #计算概率矩阵P
    tep_x3 = Z.sum(axis=0)
    tep_x4 = np.tile(tep_x3, (n, 1))
    P = Z/(tep_x4)

    tep_x5 = P * np.log(P + 1e-5)
    tep_x6 = tep_x5.sum(axis=0)
    e = - tep_x6 / math.log(n)
    d = 1-e
    tep_x7 = d.sum()
    w = d/tep_x7
    return w


def temp2(x_mat):
    n, m = np.shape(x_mat)
    tep_x1 = (x_mat * x_mat).sum(axis=0)  # 每个元素平方后按列相加
    tep_x2 = np.tile(tep_x1, (n, 1))  # 将矩阵tep_x1平铺n行
    Z = x_mat / ((tep_x2) ** 0.5)  # Z为标准化矩阵
    return Z


def temp3(answer, w):
    n, m = np.shape(answer)
    tep_max = answer.max(0)  # 得到Z中每列的最大值
    tep_min = answer.min(0)  # 每列的最小值
    tep_a = answer - np.tile(tep_max, (n, 1))  # 将tep_max向下平铺n行,并与Z中的每个对应元素做差
    tep_i = answer - np.tile(tep_min, (n, 1))  # 将tep_min向下平铺n行，并与Z中的每个对应元素做差
    D_P = (((tep_a ** 2)*w).sum(axis=1)) ** 0.5  # D+与最大值的距离向量
    D_N = (((tep_i ** 2)*w).sum(axis=1)) ** 0.5
    S = D_N / (D_P + D_N)  # 未归一化的得分
    std_S = np.around(S / S.sum(axis=0), 8).tolist()
    return std_S


def main():
    answer1 = np.loadtxt('08-result.csv', encoding='UTF-8-sig', delimiter=',', usecols=(1, 2, 3, 4))  # 推荐使用csv格式文件
    with open('08-result.csv', encoding='utf-8-sig') as f:
        # next(f)
        protein = []
        for row in csv.reader(f, skipinitialspace=True):
            protein.append(row[0])

    n, m = np.shape(answer1)
    print("共有", n, "个评价对象", m, "个评价指标")
    answer3 = temp2(answer1)  # 正向数组标准化
    w1 = entropy(answer1)  # 计算权重
    std_S = temp3(answer3, w1)  # topsis
    sorted_S = np.sort(std_S, axis=0)
    #print(answer4)

    b = np.array([[1, 1 / 3, 3, 3], [3, 1, 3, 4], [1 / 3, 1 / 3, 1, 3], [1 / 3, 1 / 4, 1 / 3, 1]])
    AHP(b).test_consist()
    w2 = AHP(b).cal_weight__by_eigenvalue_method()

    w=0.5*w2+0.5*w1
    print(w1)
    print(w2)
    print(w)

    with open('08-score.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for i in range(n):
            writer.writerow([protein[i], std_S[i]])

    max_five = heapq.nlargest(5, sorted_S)
    print(max_five)
    # print(type(max_five[0]))

    #print(type(std_S))

    max_five_index = []
    for i in range(5):
        max_five_index.append(std_S.index(np.float(max_five[i])))

    print(max_five_index)
    for i in range(5):
        print(protein[max_five_index[i]])  # 打印标准化后的得分


if __name__ == '__main__':
    main()