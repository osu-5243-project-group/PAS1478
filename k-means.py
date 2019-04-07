import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans, MiniBatchKMeans

N_CLUSTERS = 10
N_COMPONENTS = 20
MAX_K = 100
K_STEP = 5

def get_data(data_filename,labels_filename):
	data_holder=np.genfromtxt(data_filename,dtype='str',delimiter='\n')
	labels_holder=np.genfromtxt(labels_filename,dtype='str',delimiter='\n')
	labels=[]
	data=[]
	
	for index in range(0,len(labels_holder),1):
		l=labels_holder[index].split(" ")
		if(l[0]!="EMPTY"):
			data.append(data_holder[index])
			labels.append(l[0])
	return data,labels

def cluster(vector_data, labels, n_clusters):
    svd = TruncatedSVD(n_components = N_COMPONENTS, n_iter=10, random_state=42, tol=0.0)
    svd_data = svd.fit_transform(vector_data)

    estimator = MiniBatchKMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
    kmeans = estimator.fit(svd_data)
    assigned = kmeans.labels_

    h_score = metrics.homogeneity_score(labels, assigned)
    c_score = metrics.completeness_score(labels, assigned)

    return h_score, c_score, kmeans, svd_data

def plot_clustering(vector_data, labels, n_clusters):
    h_score, c_score, kmeans, svd_data = cluster(vector_data, labels, n_clusters)

    plot_svd = TruncatedSVD(n_components=2, n_iter=10, random_state=42, tol=0.0)
    plot_svd_data = plot_svd.fit_transform(svd_data)
    plot_svd_centroids = plot_svd.fit_transform(kmeans.cluster_centers_)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .01     # point in the mesh [x_min, x_max]x[y_min, y_max].
    margin = 0.1

    x_min, x_max = plot_svd_data[:, 0].min() - margin, plot_svd_data[:, 0].max() + margin
    y_min, y_max = plot_svd_data[:, 1].min() - margin, plot_svd_data[:, 1].max() + margin

    label_index = {}
    for label in kmeans.labels_:
        if label not in label_index:
            label_index[label] = len(label_index)

    cmap = plt.get_cmap('Set1')

    plt.figure(1)
    plt.clf()
    plt.scatter(plot_svd_data[:,0], plot_svd_data[:,1], s=2,
            c=kmeans.labels_,
            cmap=cmap)
    plt.scatter(plot_svd_centroids[:, 0], plot_svd_centroids[:, 1],
            marker='x', s=169, linewidths=3,
            c=np.arange(n_clusters),
            cmap=cmap, zorder=10)
    plt.title('K-means clustering with {} clusters (2D SVD-projected plot)\n'
          'Centroids are marked with a colored cross'.format(n_clusters))
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.show()

def main():
    data,labels=get_data('reduced_data/reduced_titles.out','reduced_data/places_labels.out')
    data_vectorizer = CountVectorizer()
    vector_data = data_vectorizer.fit_transform(data)

    n_labels = len(set(labels))

    k_values = []
    h_scores = []
    c_scores = []
    for i in range(1, MAX_K, K_STEP):
        h_score, c_score, kmeans, svd_data = cluster(vector_data, labels, i)
        k_values.append(i)
        h_scores.append(h_score)
        c_scores.append(c_score)

    # Plot h scores vs k
    plt.figure(1)
    plt.clf()
    plt.title('K-Means Homogeneity vs K (number of clusters)\nTitle Vectors and Places Labels')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Homogeneity')
    plt.plot(k_values, h_scores, '-', lw=2)
    plt.show()

    n_clusters = int(input('number of clusters to use for plot:'))
    plot_clustering(vector_data, labels, n_clusters)

main()