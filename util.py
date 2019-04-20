import json,pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.manifold import TSNE
from scipy.sparse import csr_matrix
import pandas as pd

def json2txt():
    with open("review_yelp_all_12992.json",'r') as load_f:
        load_dict = json.load(load_f)
        # print(load_dict)

    with open("reviewTxt.txt","w") as dump_f:
        for item in load_dict:
            dump_f.write(load_dict[item])

# 从gensim稀疏矩阵到np_array的格式转换
def gen2csr(lsi_corpus_total):
    data = []
    rows = []
    cols = []
    line_count = 0
    for line in lsi_corpus_total:  # lsi_corpus_total 是之前由gensim生成的lsi向量
        for elem in line:
            rows.append(line_count)
            cols.append(elem[0])
            data.append(elem[1])
        line_count += 1
    lsi_sparse_matrix = csr_matrix((data,(rows,cols))) # 稀疏向量
    lsi_matrix = lsi_sparse_matrix.toarray()  # 密集向量
    with open("./data/matrix.txt", 'wb') as f:
        pickle.dump(lsi_matrix,f)
    return lsi_matrix

# 查看特征
def getFeature():
    with open("feature8.txt",'rb') as f:
        d = pickle.load(f)
    for i in range(100):
        print(d[i])

def featureFilter(num = 500):
    with open("./data/matrix.txt",'rb') as f:
        rawData = pickle.load(f)
    
    rawData = pd.DataFrame(rawData)
    l = len(rawData)
    m_filter = [False for _ in range(l)]

    for i in range(8):
        selected = rawData.nlargest(num,i).index
        for j in selected:
            m_filter[j] = True
    
    rawData = rawData[m_filter]
    print(len(rawData))

    with open("./data/filter_matrix.txt",'wb') as f:
        pickle.dump(rawData,f)

    


def dimentionRed(numdim = 3):
    # 读取原始数据
    with open("./data/filter_matrix.txt",'rb') as f:
        rawData = pickle.load(f)
    rawData = pd.DataFrame(rawData)

    # 降维
    tsne = TSNE(n_components=numdim)
    tsne.fit_transform(rawData) #进行数据降维       
    rawData = pd.DataFrame(tsne.embedding_, index = rawData.index) #转换数据格式
    with open("./data/filter_matrix3d.txt",'wb') as f:
        pickle.dump(rawData,f)


def plotResult(route=None,type="original",version=[0]):
    
    # 读取降维后的原始数据
    # with open("./data/filter_matrix3d.txt",'rb') as f:
    #     rawData = pickle.load(f)
    rawData = pd.read_pickle("./data/filter_matrix3d.txt")
    rawData = pd.DataFrame(rawData)

    if route is None:
        clusters = [0 for _ in range(len(rawData))]
    else:
        with open(route,'rb') as f:
            clusters = pickle.load(f)
    numTypes = len(set(clusters)) - (1 if -1 in clusters else 0)

    rawData['tag'] = clusters

    fig = plt.figure()
    ax = Axes3D(fig)
    colors = ['blue','green','purple','red','yellow','cyan','m']
    # cluster core
    for i in range(numTypes):
        index = i
        x = rawData[rawData['tag']==index][0]
        y = rawData[rawData['tag']==index][1]
        z = rawData[rawData['tag']==index][2]
        ax.scatter(x,y,z,c=colors[i], alpha=0.5,s = 150)
        # ax11.annotate(i,(rawData[0][index],rawData[1][index]), fontsize=16,fontweight="bold")

    # cluster halo
    noise = rawData[rawData['tag']==-1]
    ax.scatter(noise[0],noise[1],noise[2],c='black', alpha=0.5,s = 150)

    version[0] += 1
    plt.show()
    # plt.savefig("./img/"+type+str(version[0])+".jpg")

if __name__ == "__main__":

    # featureFilter(num=200) 
    # dimentionRed()
    # plotResult("./tags/spectral_tag.txt")
    


    with open("./data/filter_matrix.txt",'rb') as f:
        rdata = pickle.load(f)

    with open("./data/review_yelp_all_12992.json",'r') as load_f:
        load_dict = json.load(load_f)

    ids = rdata.index
    count = 0
    idList = []
    contentList = []
    for key in load_dict.keys():
        if count in ids:
            idList.append(key)
            contentList.append(load_dict[key])
        count += 1

    dir = './tags/'
    res = pd.DataFrame(index=rdata.index)
    with open(dir+'diy_tag.txt','rb') as f:
        diy = pickle.load(f)
    with open(dir+'em_tag.txt','rb') as f:
        em = pickle.load(f)
    with open(dir+'dbscan_tag.txt','rb') as f:
        dbscan = pickle.load(f)
    with open(dir+'hierarch_tag.txt','rb') as f:
        hierarch = pickle.load(f)
    with open(dir+'kmeans_tag.txt','rb') as f:
        kmeans = pickle.load(f)
    with open(dir+'spectral_tag.txt','rb') as f:
        spectral = pickle.load(f)

    res['sci'] = diy
    res['em'] = em
    res['dbscan'] = dbscan
    res['hierarch'] = hierarch
    res['kmeans'] = kmeans
    res['spectral'] = spectral
    res['content'] = contentList
    res['id'] = idList

    res.to_csv("./data/res.csv")


    