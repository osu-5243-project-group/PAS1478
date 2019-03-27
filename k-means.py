import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans, MiniBatchKMeans

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

def main():
    data,labels=get_data('reduced_data/reduced_titles.out','reduced_data/topics_labels.out')
    data_vectorizer = CountVectorizer()
    vector_data = data_vectorizer.fit_transform(data)

    n_labels = len(set(labels))

    svd = TruncatedSVD(n_components = 2, n_iter=10, random_state=42, tol=0.0)
    svd_data = svd.fit_transform(vector_data)

    estimator = MiniBatchKMeans(init='k-means++', n_clusters=n_labels, n_init=10)
    kmeans = estimator.fit(svd_data)
    centroids = kmeans.cluster_centers_

    # SVD for 2D plotting
    #svd = TruncatedSVD(n_components = 2)
    #svd_data = svd.fit_transform(vector_data)
    svd_centroids = centroids #svd.transform(centroids)
    
    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .01     # point in the mesh [x_min, x_max]x[y_min, y_max].
    margin = 0.1

    x_min, x_max = svd_data[:, 0].min() - margin, svd_data[:, 0].max() + margin
    y_min, y_max = svd_data[:, 1].min() - margin, svd_data[:, 1].max() + margin
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    #Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    label_index = {}
    for label in kmeans.labels_:
        if label not in label_index:
            label_index[label] = len(label_index)

    cmap = plt.get_cmap('Set1')

    # Put the result into a color plot
    #Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    # plt.imshow(Z, interpolation='nearest',
    #        extent=(xx.min(), xx.max(), yy.min(), yy.max()),
    #        cmap=plt.cm.Paired,
    #        aspect='auto', origin='lower')
    plt.scatter(svd_data[:,0], svd_data[:,1], s=2,
            c=kmeans.labels_,
            cmap=cmap)
    plt.scatter(svd_centroids[:, 0], svd_centroids[:, 1],
            marker='x', s=169, linewidths=3,
            c=np.arange(n_labels),
            cmap=cmap, zorder=10)
    plt.title('K-means clustering with {} clusters (SVD-projected plot)\n'
          'Centroids are marked with colored cross'.format(n_labels))
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()

main()