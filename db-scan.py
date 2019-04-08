import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

N_CLUSTERS = 10
N_COMPONENTS = 18
MAX_SAMP = 400
SAMP_STEP = 33

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

def cluster(vector_data, labels, min_samples):
    # svd = TruncatedSVD(n_components = N_COMPONENTS, n_iter=10, random_state=42, tol=0.0)
    # svd_data = svd.fit_transform(vector_data)
    svd_data=vector_data
    estimator = DBSCAN(eps=0.5, min_samples=min_samples)
    dbscan = estimator.fit(svd_data)
    assigned = dbscan.labels_
    h_score = metrics.homogeneity_score(labels, assigned)
    c_score = metrics.completeness_score(labels, assigned)
    return h_score, c_score, dbscan, svd_data

def plot_clustering(vector_data, labels, min_samples):
    h_score, c_score, dbscan, s_data = cluster(vector_data, labels, min_samples)

    plot_svd = TruncatedSVD(n_components=2, n_iter=10, random_state=42, tol=0.0)
    plot_svd_data = plot_svd.fit_transform(s_data)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .01     # point in the mesh [x_min, x_max]x[y_min, y_max].
    margin = 0.1

    x_min, x_max = plot_svd_data[:, 0].min() - margin, plot_svd_data[:, 0].max() + margin
    y_min, y_max = plot_svd_data[:, 1].min() - margin, plot_svd_data[:, 1].max() + margin

    label_index = {}
    for label in dbscan.labels_:
        if label not in label_index:
            label_index[label] = len(label_index)

    cmap = plt.get_cmap('Set1')

    plt.figure(1)
    plt.clf()
    plt.scatter(plot_svd_data[:,0], plot_svd_data[:,1], s=2,
        c=dbscan.labels_,
        cmap=cmap)
    plt.title('DBSCAN clustering with {} min-samples (2D SVD-projected plot)\n'
    'Pricipal components are marked with a colored cross'.format(min_samples))
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.show()

def vector_data(filename,labels_filename):
    data,labels=get_data(filename,labels_filename)
    data_vectorizer = CountVectorizer()
    vector_data = data_vectorizer.fit_transform(data)
    if(vector_data.shape[1]<N_COMPONENTS):
        width=N_COMPONENTS-vector_data.shape[1]
        vector_data=np.pad(vector_data,(0,width),'constant',constant_values=(0))

    svd = TruncatedSVD(n_components = N_COMPONENTS, n_iter=10, random_state=42, tol=0.0)
    svd_data = svd.fit_transform(vector_data)
    return svd_data,labels

def combine_vectors(weights,unique,bodies,orgs,people,exchanges,dates):
    weighted_unique=np.multiply(0.1,unique)
    weighted_people=np.multiply(0.1,people)
    weighted_orgs=np.multiply(0.2,orgs)
    weighted_dates=np.multiply(0.3,dates)
    weighted_exchanges=np.multiply(0.7,exchanges)
    weighted_bodies=np.multiply(0.5,bodies)


    together_1=np.add(weighted_people,weighted_orgs)
    together_2=np.add(weighted_dates,weighted_exchanges)
    together_3=np.add(together_1,together_2)

    group_together_1=np.add(weighted_unique,weighted_bodies)
    group_together_2=np.multiply(0.4,together_3)

    together=np.add(group_together_1,group_together_2)
    return together


def main():

    title_vectors,title_labels=vector_data('reduced_data/reduced_titles.out','reduced_data/topics_labels.out')
    bodies_vectors,bodies_labels=vector_data('reduced_data/reduced_bodies.out','reduced_data/topics_labels.out')
    orgs_vectors,orgs_labels=vector_data('reduced_data/reduced_orgs.out','reduced_data/topics_labels.out')
    people_vectors,people_labels=vector_data('reduced_data/reduced_people.out','reduced_data/topics_labels.out')
    exchanges_vectors,exchanges_labels=vector_data('reduced_data/reduced_exchanges.out','reduced_data/topics_labels.out')
    dates_vectors,dates_labels=vector_data('reduced_data/reduced_dates.out','reduced_data/topics_labels.out')
    weights=[]
    combined_svd_vector=combine_vectors(weights,title_vectors,bodies_vectors,orgs_vectors,people_vectors,exchanges_vectors,dates_vectors)
    # data_vectorizer = CountVectorizer()
    # vector_data = data_vectorizer.fit_transform(data)

    n_labels = len(set(title_labels))

    k_values = []
    h_scores = []
    c_scores = []
    for i in range(5, MAX_SAMP, SAMP_STEP):
        h_score, c_score, dbscan, svd_data = cluster(combined_svd_vector, title_labels, i)
        k_values.append(i)
        h_scores.append(h_score)
        c_scores.append(c_score)

    # Plot h scores vs k
    plt.figure(1)
    plt.clf()
    plt.title('DBSCAN Homogeneity vs Min-Samples\nTitle Vectors and Topics Labels')
    plt.xlabel('Min Sample Count')
    plt.ylabel('Homogeneity')
    plt.plot(k_values, h_scores, '-', lw=2)
    plt.show()

    n_clusters = int(input('min-samples to use for plot:'))
    plot_clustering(combined_svd_vector, title_labels, n_clusters)

main()
