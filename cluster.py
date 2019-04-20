import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN 
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
# from sklearn.mixture import GaussianMixture
from diyModel import DiyModel


from sklearn import metrics
import matplotlib.pyplot as plt

from util import gen2csr,plotResult
import pickle,os,json


# 画图寻找最优聚类个数
def plot_cluster_measure(matrix,av_tries = 3,start=2,end=10,type='kmeans'):
    
    measurement = [[],[],[]]
    for num in range(start,end+1):

        num_clusters = num
        # elbow = 0
        silhouette_score = 0
        calin_score = 0

        for _ in range(av_tries):
            if type == 'kmeans':
                km_cluster = KMeans(n_clusters=num_clusters)
            elif type == 'em':
                km_cluster = GaussianMixture(n_components=num_clusters)
            elif type == 'hierarch':
                km_cluster =AgglomerativeClustering(n_clusters=num_clusters)
            else:
                continue

            #返回各自文本的所被分配到的类索引
            kmeans_model = km_cluster.fit(matrix)       
            labels = kmeans_model.predict(matrix)

            # elbow += kmeans_model.inertia_
            silhouette_score += metrics.silhouette_score(matrix, labels, metric='euclidean')
            calin_score += metrics.calinski_harabaz_score(matrix, labels)

        # measurement[0].append(elbow/av_tries)
        measurement[1].append(silhouette_score/av_tries)
        measurement[2].append(calin_score/av_tries)

    num_clusters = list(range(start,end+1))
    plt.figure()
    # plt.plot(num_clusters,measurement[0])
    # plt.savefig("./img/clusterNum_elbow.png")
    # plt.clf()
    plt.plot(num_clusters,measurement[1])
    plt.savefig("./img/clusterNum_silhouette.png")
    plt.clf()
    plt.plot(num_clusters,measurement[2])
    plt.savefig("./img/clusterNum_calin.png")

def write_tag(tags,catag):
    with open("./tags/"+catag+'_tag.txt', 'wb') as f:
        pickle.dump(tags,f)

def print_res(labels,silFactor,cFactor=0):
    type = len(set(labels)) - (1 if -1 in labels else 0)
    for i in range(type):
        print(str(i) + " : " + str(len(labels[labels==i])))
    print("s_score: " + str(silFactor))
    print("c_score: " + str(cFactor))

# def sample():
#     data = np.random.rand(100, 3) #生成一个随机数据，样本大小为100, 特征数为3

#     #假如我要构造一个聚类数为3的聚类器
#     estimator = KMeans(n_clusters=3)#构造聚类器
#     estimator.fit(data)#聚类
#     label_pred = estimator.labels_ #获取聚类标签
#     centroids = estimator.cluster_centers_ #获取聚类中心
#     inertia = estimator.inertia_ # 获取聚类准则的总和

if __name__ == "__main__":
    # 读取格式转换后的特征
    # with open("./data/filter_matrix.txt",'rb') as f:
    #     matrix = pickle.load(f)
    matrix = pd.read_pickle('./data/filter_matrix.txt')
    
    # 格式转换
    # matrix = gen2csr(d) 

    # 聚类
    ####################################
    '''KMEANS'''

    # plot_kmeans_measure(matrix,av_tries=5,start=2,end=10)s
    # num_clusters = 3
    # km_cluster = KMeans(n_clusters=num_clusters,n_jobs=-1)

    # kmeans_model = km_cluster.fit(matrix)       
    # labels = kmeans_model.labels_
    # silhouette_score = metrics.silhouette_score(matrix, labels, metric='euclidean')
    
    # print_res(labels,silhouette_score)

    # write_tag(labels,catag="kmeans")

    ####################################
    '''DBSCAN'''
    
    # db_cluster = DBSCAN(eps=0.2, min_samples=20)
    # db_model = db_cluster.fit(matrix)

    # labels = db_model.labels_
    # silhouette_score = metrics.silhouette_score(matrix, labels, metric='euclidean')

    # f = open('result.txt','w')

    # for num in [0.15,0.2,0.25]:

    #     eps = num

    #     for samples in range(10,35,5):
    #         f.write("\n\neps = "+str(eps))
    #         f.write("\nmin_samples = "+str(samples))

    #         db_cluster = DBSCAN(eps=eps, min_samples=samples)
    #         db_model = db_cluster.fit(matrix)

    #         labels = db_model.labels_
    #         silhouette_score = metrics.silhouette_score(matrix, labels, metric='euclidean')
    #         calin_score = metrics.calinski_harabaz_score(matrix, labels)

    #         type = len(set(labels)) - (1 if -1 in labels else 0)
    #         for i in range(type):
    #             f.write("\n"+str(i) + " : " + str(len(labels[labels==i])))
    #         f.write("\ns_score: " + str(silhouette_score))
    #         f.write("\nc_score: " + str(calin_score))
    
    # f.close()
    # write_tag(labels,"dbscan")
    
    
    ####################################
    '''Spectral'''
    # labels = SpectralClustering(n_clusters=3, gamma=0.15).fit_predict(matrix)

    # silhouette_score = metrics.silhouette_score(matrix, labels, metric='euclidean')

    # write_tag(labels,"spectral")

    # f = open('result2.txt','w')

    # for num in [0.1,0.12,0.15,0.17,0.19]:

    #     eps = num

    #     for samples in range(3,5):
    #         f.write("\n\ngamma = "+str(eps))
    #         f.write("\n_clusters = "+str(samples))

    #         labels = SpectralClustering(n_clusters=samples, gamma=eps).fit_predict(matrix)

    #         silhouette_score = metrics.silhouette_score(matrix, labels, metric='euclidean')
    #         calin_score = metrics.calinski_harabaz_score(matrix, labels)

    #         type = len(set(labels)) - (1 if -1 in labels else 0)
    #         for i in range(type):
    #             f.write("\n"+str(i) + " : " + str(len(labels[labels==i])))
    #         f.write("\ns_score: " + str(silhouette_score))
    #         f.write("\nc_score: " + str(calin_score))
    
    # f.close()

    ####################################
    '''Hierarchical'''

    # labels = AgglomerativeClustering(linkage='average',n_clusters=3).fit_predict(matrix)

    # silhouette_score = metrics.silhouette_score(matrix, labels, metric='euclidean')
    # calin_score = metrics.calinski_harabaz_score(matrix, labels)

    # write_tag(labels,"hierarch")

    ####################################
    '''EM-GMM'''

    # gmmModel = GaussianMixture(n_components=3,max_iter=500)
    # gmmModel.fit(matrix)
    # labels = gmmModel.predict(matrix)
    # silhouette_score = metrics.silhouette_score(matrix, labels, metric='euclidean')
    # calin_score = metrics.calinski_harabaz_score(matrix, labels)

    # write_tag(labels,"em")

    # plot_cluster_measure(matrix,type='em')


    ####################################
    '''Task1'''

    rawData = pd.DataFrame(matrix)
    labels = DiyModel(rawData,dcPercent=0.02,n_cluster=3)

    labels = labels.values
    silhouette_score = metrics.silhouette_score(matrix, labels, metric='euclidean')
    # calin_score = metrics.calinski_harabaz_score(matrix, labels)

    write_tag(labels,"diy")


    ####################################
    # 可视化、输出结果
    print_res(labels,silhouette_score) 

    plotResult("./tags/diy_tag.txt")