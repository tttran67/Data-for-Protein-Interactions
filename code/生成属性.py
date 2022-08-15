import csv
import pandas as pd
path = '2022-08-08.csv'

# 1.计算 score sum
scoresum = dict()
# 计算每个结点相连的所有度
with open(path, encoding='utf-8-sig') as f:
    next(f)
    for row in csv.reader(f, skipinitialspace=True):
        if(row[0] == row[3]): # 去掉自相连
            continue
        if row[0] not in scoresum:
            scoresum.update({row[0]: 0})
        if row[3] not in scoresum:
            scoresum.update({row[3]: 0})
        scoresum[row[0]] += float(row[6])
        scoresum[row[3]] += float(row[6])

# 2.计算 SOECC
connect = dict()
with open(path, encoding='utf-8-sig') as f:
    next(f)
    for row in csv.reader(f, skipinitialspace=True):
        if(row[0] == row[3]): # 去掉自相连
            continue
        if row[0] not in connect:
            connect.update({row[0]: list()})
        connect[row[0]].append(row[3])
        if row[3] not in connect:
            connect.update({row[3]: list()})
        connect[row[3]].append(row[0])

# 计算每个结点的SOECC
valuedict = dict()
for c in connect:
    valuedict.update({c : 0})
for pro1 in connect:
    for pro2 in connect[pro1]:
        tmp = 0
        if(pro1 == pro2):
            continue
        else:
            for pro3 in connect[pro1]:
                if(pro2 == pro3):
                    continue
                else:
                    if(pro3 in connect[pro2]):
                        tmp += 1
            if tmp != 0:
                tmp /= min(len(connect[pro1]) - 1, len(connect[pro2]) - 1)
            valuedict[pro1] += tmp

# 3.计算种群间指标
# variety 表示某结点连接的不同种群数
# outedge 表示某结点向外相连的所有边
variety = dict()
outedge = dict()
# 计算每个节点向外相连的个数
with open(path, encoding='utf-8-sig') as f:
    next(f)
    for row in csv.reader(f, skipinitialspace=True):
        if row[0] == row[3]: # 去掉自相连
            continue
        # 处理variety
        if row[0] not in variety:
            variety.update({row[0]: list()})
        if row[3] not in variety:
            variety.update({row[3]: list()})
        # 处理outedge
        if row[0] not in outedge:
            outedge.update({row[0]: list()})
        if row[3] not in outedge:
            outedge.update({row[3]: list()})

        if row[2] == row[5]: # 只找不同类的
            continue
        if row[5] not in variety[row[0]]:
            variety[row[0]].append(row[5])
        if row[2] not in variety[row[3]]:
            variety[row[3]].append(row[2])

        outedge[row[0]].append(row[3])

        outedge[row[3]].append(row[0])

# 4.以矩阵形式写入csv文件
with open(r'08-result.csv', 'a', encoding='utf-8', newline="") as f:
    writer = csv.writer(f)
    for s in scoresum:
        writer.writerow([s,scoresum[s],valuedict[s],len(variety[s]),len(outedge[s])])

