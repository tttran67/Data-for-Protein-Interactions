# neural network definition
import numpy
import scipy.special

# 实现任意输入节点、隐藏节点、输出节点的神经网络
# neural network definition
class neuralNetwork:

    # initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # set number of nodes in each input,hidden,output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate
        # link weight matrices,wih(winput_hidden) and who(whidden_output)
        # weights inside the arrays are w_i_j,where link is from node i to node j in the next layer
        # w11 w21
        # w12 w22 etc
        # self.wih = (numpy.random.random(self.hnodes,self.inodes)-0.5)
        # self.who = (numpy.random.random(self.onodes,self.hnodes)-0.5)
        # 使用正态概率分布采样权重，均值为0，标准方差为节点传入链接数目的开方(1/根号下链接传入数目)
        self.wih = (numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes)))
        self.who = (numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes)))
        self.activation_funcation = lambda x: scipy.special.expit(x)

    # train the neural network
    def train(self, inputs_list, targets_list):
        # 首先做query，计算输出
        inputs = numpy.array(inputs_list, ndmin=2).T
        targes = numpy.array(targets_list, ndmin=2).T# 和query的区别
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_funcation(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_ouputs = self.activation_funcation(final_inputs)

        # 然后对比输出和所需输出，用差值指导网络权重更新
        output_errors = targes - final_ouputs
        hidden_errors = numpy.dot(self.who.T, output_errors)
        self.who += self.lr * numpy.dot((output_errors * final_ouputs * (1.0 - final_ouputs)),
                                        numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                        numpy.transpose(inputs))


    # query the neural network
    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_funcation(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_ouputs = self.activation_funcation(final_inputs)
        return final_ouputs

# 每层3个节点，学习率为0.3
# number of input ,hidden and output nodes
input_nodes = 4
hidden_nodes = 4
output_nodes = 1

# learning rate is 0.3
learning_rate = 0.3

# create instance of neural network
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

import csv
import pandas as pd

traindata = dict()
# 计算每个结点相连的所有度
with open('result.csv', encoding='utf-8-sig') as f:
    for row in csv.reader(f, skipinitialspace=True):
        if row[0] not in traindata:
            traindata.update({row[0]: list()})
        traindata[row[0]].append(row[1])
        traindata[row[0]].append(row[2])
        traindata[row[0]].append(row[3])
        traindata[row[0]].append(row[4])

res = dict()
with open('score.csv', encoding='utf-8-sig') as f:
    for row in csv.reader(f, skipinitialspace=True):
        res.update({row[0]: row[1]})

# 训练
for t in traindata:
    result = list()
    res[t] = float(res[t])
    result.append(res[t])
    n.train([float(traindata[t][0]), float(traindata[t][1]), float(traindata[t][2]), float(traindata[t][3])], result)

# 测试
# 获取测试集
test = dict()
with open('08-result.csv', encoding='utf-8-sig') as f:
    for row in csv.reader(f, skipinitialspace=True):
        test.update({row[0]: list()})
        test[row[0]].append(row[1])
        test[row[0]].append(row[2])
        test[row[0]].append(row[3])
        test[row[0]].append(row[4])

with open(r'predict.csv', 'a', encoding='utf-8', newline="") as f:
    writer = csv.writer(f)
    for t in test:
        writer.writerow([t, float(test[t][0]), float(test[t][1]), float(test[t][2]), float(test[t][3]), n.query([float(test[t][0]), float(test[t][1]), float(test[t][2]), float(test[t][3])])])
