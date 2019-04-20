import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

def readData(fileName):
    csvData = pd.read_csv(fileName,header=None)  # 读取训练数据
    # csvData.plot.scatter(x=0, y=1)
    return csvData

def computeDistance(item1,item2):
    res = 0
    for i in range(len(item1)):
        res += (item1[i]-item2[i])**2
    return np.sqrt(res)
    # return res

# 计算距离矩阵
def distanceMatrix(rawdata,disFun=computeDistance):
    l = len(rawdata)
    res = [[0 for _ in range(l)] for _ in range(l)]
    disList = []
    for i in range(l):
        for j in range(i+1,l):
            tmp = disFun(rawdata.iloc[i],rawdata.iloc[j])
            res[i][j] = res[j][i] = tmp 
            disList.append(tmp)
    disList.sort()
    return pd.DataFrame(res),pd.Series(disList)

# 二分查找确定dc
def bs_computeDensity(dmatrix):
    l = len(dmatrix)
    left = 0
    right = dmatrix.iloc[random.randint(0,l-1)].median()    
    while(True):
        dc = (left + right)/2
        density = pd.Series([0 for _ in range(l)])
        for i in range(l):
            tmp = dmatrix.iloc[i]
            density[i] = len(tmp[tmp <= dc])
        averageNeighbors = density.mean()
        if averageNeighbors < 0.01*l:
            left = dc
        elif averageNeighbors > 0.02*l:
            right = dc
        else:               
            break
        print("dc: " + str(dc))
        print("averageNeighbors: " + str(averageNeighbors))
    print("dc found: " + str(dc))
    print("averageNeighbors: " + str(averageNeighbors))
    return density

# 直接确定dc,截断距离计算密度
def computeDensity(dmatrix,dlist,dcPercent=0.02):
    l = len(dmatrix)  
    position = int(len(dlist)*dcPercent)
    dc = dlist.iloc[position,0]
    density = pd.Series([0 for _ in range(l)])
    for i in range(l):
        tmp = dmatrix.iloc[i]
        density[i] = len(tmp[tmp <= dc])
    averageNeighbors = density.mean()
    print("dc found: " + str(dc))
    print("averageNeighbors: " + str(averageNeighbors))
    return density,dc

# 直接确定dc,高斯核函数计算密度
def gs_computeDensity(dmatrix,dlist,dcPercent=0.02):
    l = len(dmatrix)  
    position = int(len(dlist)*dcPercent)
    dc = dlist.iloc[position,0]
    density = pd.Series([0.0 for _ in range(l)])
    for i in range(l):
        tmp = dmatrix.iloc[i]
        # -1 是为了减去矩阵中对角线上的0元素产生的1
        density[i] = tmp.map(lambda x: np.exp(-(x/dc)*(x/dc))).sum()-1
    averageNeighbors = density.mean()

    print("dc found: " + str(dc))
    print("averageNeighbors: " + str(averageNeighbors))
    return density,dc

# compute min distance to higher density
def computeMDTH(dmatrix,density):
    l = len(dmatrix)
    MDTH = [0 for _ in range(l)]
    for i in range(l):
        currentDistance = dmatrix.iloc[i,:]
        qualifiedIndex = density[(density >= density[i]) & (density.index != i) ].index
        qualifiedDistance = currentDistance[qualifiedIndex]
        if len(qualifiedDistance) == 0:
            MDTH[i] = currentDistance.max()
        else:
            MDTH[i] = qualifiedDistance.min()
    return pd.Series(MDTH)

def assignTag(dmatrix,rawData,clusters):
    l = len(dmatrix)
    tag = [0 for _ in range(l)]
    count = 0 
    id2tag = {}
    for i in range(l):
        currentDistance = dmatrix.iloc[i,:]
        distances2Centers = currentDistance[clusters]
        tmp = distances2Centers.idxmin()
        if not tmp in id2tag:
            id2tag[tmp] = count
            count += 1
        tag[i] = id2tag[tmp]
    rawData['tag'] = tag

def findNoise(dmatrix,density,rawData,clusters,dc):
    for i in clusters:
        clusterIndex = i
        clusterMem = rawData[rawData['tag'] == clusterIndex].index
        border = []
        for mem in clusterMem:
            currentDistance = dmatrix.iloc[mem]
            for i in range(len(currentDistance)):
                if not i in clusterMem and currentDistance[i] < dc:
                    border.append(mem)
                    break               
        maxBorderDensity = density[border].max()
        for mem in clusterMem:
            if density[mem] <= maxBorderDensity:
                rawData.ix[mem,'tag'] = -1

def plotResult(density,mdth,rawData):
    f,(ax11, ax12) = plt.subplots(1, 2)

    # ax12.scatter(rawData[0],rawData[1],s=density*5)
    # ax11.scatter((density*mdth).index,(density*mdth).sort_values())
    ax12.scatter(density,mdth)
    colors = ['blue','green','purple','red','yellow','cyan','m']
    # cluster core
    for i in range(7):
        index = clusters[i]
        x = rawData[rawData['tag']==index][0]
        y = rawData[rawData['tag']==index][1]
        ax11.scatter(x,y,c=colors[i], alpha=0.5,s = 150)
        ax11.annotate(i,(rawData[0][index],rawData[1][index]), fontsize=16,fontweight="bold")
        ax12.annotate(i,(density[index],mdth[index]), fontsize=16)
    # cluster halo
    noise = rawData[rawData['tag']==-1]
    ax11.scatter(noise[0],noise[1],c='black', alpha=0.5,s = 150)
    plt.show()

def DiyModel(rawData,dcPercent=0.02,n_cluster=4):
    # dmatrix,dlist = distanceMatrix(rawData,computeDistance)
    # dmatrix.to_csv("dmatrix1.csv")
    # dlist.to_csv("dlist1.csv")
    dmatrix = pd.read_csv("dmatrix1.csv",index_col=0,header=0)
    dlist = pd.read_csv("dlist1.csv",index_col=0,header=0)

    density,dc = gs_computeDensity(dmatrix,dlist=dlist,dcPercent=dcPercent)
    mdth = computeMDTH(dmatrix,density)

    # plt.figure()
    plt.scatter(density,mdth)
    plt.savefig("./img/decision_tree.png")

    clusters = (density*mdth).sort_values(ascending=False)[:n_cluster].index
    assignTag(dmatrix,rawData,clusters)
    findNoise(dmatrix,density,rawData,clusters,dc)

    return rawData['tag']

if __name__ == "__main__":
    # 读取数据
    rawData = readData("test.csv")

    # 计算距离矩阵
    # dmatrix,dlist = distanceMatrix(rawData,computeDistance)
    # dmatrix.to_csv("dmatrix1.csv")
    # dlist.to_csv("dlist1.csv")
    dmatrix = pd.read_csv("dmatrix1.csv",index_col=0,header=0)
    dlist = pd.read_csv("dlist1.csv",index_col=0,header=0)

    # 计算局部密度和最近距离
    # density,dc = computeDensity(dmatrix,dlist=dlist,dcPercent=0.02)
    density,dc = gs_computeDensity(dmatrix,dlist=dlist,dcPercent=0.02)
    mdth = computeMDTH(dmatrix,density)

    # 确定聚类中心、生成聚类结果
    clusters = (density*(mdth**2)).sort_values(ascending=False)[:7].index
    assignTag(dmatrix,rawData,clusters)
    findNoise(dmatrix,density,rawData,clusters,dc)

    # 结果可视化
    plotResult(density,mdth,rawData)
