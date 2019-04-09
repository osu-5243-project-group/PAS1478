import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

SKIP_ELBOW=True
PLOT_COMPONENTS = 2
N_CLUSTERS = 10
N_COMPONENTS = 18
MAX_SAMP = 400
MIN_SAMP = 5
STEP_SAMP = 33
MAX_EPS = 1.0
MIN_EPS = 0.1
STEP_EPS = 0.1

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

def cluster(vector_data, labels, min_samples, eps):
    # svd = TruncatedSVD(n_components = N_COMPONENTS, n_iter=10, random_state=42, tol=0.0)
    # svd_data = svd.fit_transform(vector_data)
    svd_data=vector_data
    estimator = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    dbscan = estimator.fit(svd_data)
    assigned = dbscan.labels_
    h_score = metrics.homogeneity_score(labels, assigned)
    c_score = metrics.completeness_score(labels, assigned)
    return h_score, c_score, dbscan, svd_data

def plot_clustering(vector_data, labels, min_samples, eps):
    h_score, c_score, dbscan, s_data = cluster(vector_data, labels, min_samples, eps)

    print('h score: ', h_score)
    print('c score: ', c_score)

    plot_svd = TruncatedSVD(n_components=PLOT_COMPONENTS, n_iter=10, random_state=42, tol=0.0)
    plot_svd_data = plot_svd.fit_transform(s_data)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .01     # point in the mesh [x_min, x_max]x[y_min, y_max].
    margin = 0.1

    x_min, x_max = plot_svd_data[:, 0].min() - margin, plot_svd_data[:, 0].max() + margin
    y_min, y_max = plot_svd_data[:, 1].min() - margin, plot_svd_data[:, 1].max() + margin
    #z_min, z_max = plot_svd_data[:, 2].min() - margin, plot_svd_data[:, 2].max() + margin

    label_index = {}
    for label in dbscan.labels_:
        if label not in label_index:
            label_index[label] = len(label_index)

    cmap = plt.get_cmap('Set1')

    fig = plt.figure(1)
    plt.clf()
    ####
    # ax=fig.add_subplot(111,projection='3d')
    # ax.scatter(plot_svd_data[:,0], plot_svd_data[:,1], plot_svd_data[:,2], s=2,
    #     c=dbscan.labels_,
    #     cmap=cmap)
    # ax.set_title('DBSCAN clustering (3D SVD-projected plot)\n{} min-samples and {} epsilon radius'.format(min_samples, eps))
    # ax.set_xlim(x_min, x_max)
    # ax.set_ylim(y_min, y_max)
    #ax.set_zlim(z_min, z_max)
    #####

    plt.scatter(plot_svd_data[:,0], plot_svd_data[:,1], s=2,
        c=dbscan.labels_,
        cmap=cmap)
    plt.title('DBSCAN clustering (2D SVD-projected plot)\n{} min-samples and {} epsilon radius'.format(min_samples, eps))
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    #####

    plt.show()

def vector_data(filename,labels_filename):
	data,labels=get_data(filename,labels_filename)
	data_vectorizer = CountVectorizer()
	vector_data = data_vectorizer.fit_transform(data)
	svd = TruncatedSVD(n_components = N_COMPONENTS, n_iter=10, random_state=42, tol=0.0)
	svd_data = svd.fit_transform(vector_data)
	return svd_data,labels

def combine_vectors(type,unique,bodies,orgs,people,exchanges,dates):
	if(type=='places'):
		weighted_unique=np.multiply(0.1,unique)
		weighted_people=np.multiply(0.9,people)
		weighted_orgs=np.multiply(0.03,orgs)
		weighted_dates=np.multiply(0.03,dates)
		weighted_exchanges=np.multiply(0.03,exchanges)
		weighted_bodies=np.multiply(0.5,bodies)


		together_1=np.add(weighted_people,weighted_orgs)
		together_2=np.add(weighted_dates,weighted_exchanges)
		together_3=np.add(together_1,together_2)

		group_together_1=np.add(weighted_unique,weighted_bodies)
		group_together_2=np.multiply(0.4,together_3)

		together=np.add(group_together_1,group_together_2)
		return together
	else:
		weighted_unique=np.multiply(0.5,unique)
		weighted_people=np.multiply(0.25,people)
		weighted_orgs=np.multiply(0.05,orgs)
		weighted_dates=np.multiply(0.25,dates)
		weighted_exchanges=np.multiply(0.25,exchanges)
		weighted_bodies=np.multiply(0.3,bodies)


		together_1=np.add(weighted_people,weighted_orgs)
		together_2=np.add(weighted_dates,weighted_exchanges)
		together_3=np.add(together_1,together_2)

		group_together_1=np.add(weighted_unique,weighted_bodies)
		group_together_2=np.multiply(0.2,together_3)

		together=np.add(group_together_1,group_together_2)
		return together


def main():

    type='topics'
    if(type=='topics'):
        title_vectors,title_labels=vector_data('reduced_data/reduced_titles.out','reduced_data/topics_labels.out')
        bodies_vectors,bodies_labels=vector_data('reduced_data/reduced_bodies.out','reduced_data/topics_labels.out')
        orgs_vectors,orgs_labels=vector_data('reduced_data/reduced_orgs.out','reduced_data/topics_labels.out')
        people_vectors,people_labels=vector_data('reduced_data/reduced_people.out','reduced_data/topics_labels.out')
        exchanges_vectors,exchanges_labels=vector_data('reduced_data/reduced_exchanges.out','reduced_data/topics_labels.out')
        dates_vectors,dates_labels=vector_data('reduced_data/reduced_dates.out','reduced_data/topics_labels.out')
        combined_svd_vector=combine_vectors('topics',title_vectors,bodies_vectors,orgs_vectors,people_vectors,exchanges_vectors,dates_vectors)
        n_labels = len(set(title_labels))
        labels=title_labels
    else:
        dateline_vectors,dateline_labels=vector_data('reduced_data/reduced_titles.out','reduced_data/places_labels.out')
        bodies_vectors,bodies_labels=vector_data('reduced_data/reduced_bodies.out','reduced_data/places_labels.out')
        orgs_vectors,orgs_labels=vector_data('reduced_data/reduced_orgs.out','reduced_data/places_labels.out')
        people_vectors,people_labels=vector_data('reduced_data/reduced_people.out','reduced_data/places_labels.out')
        exchanges_vectors,exchanges_labels=vector_data('reduced_data/reduced_exchanges.out','reduced_data/places_labels.out')
        dates_vectors,dates_labels=vector_data('reduced_data/reduced_dates.out','reduced_data/places_labels.out')
        combined_svd_vector=combine_vectors('places',dateline_vectors,bodies_vectors,orgs_vectors,people_vectors,exchanges_vectors,dates_vectors)
        n_labels = len(set(dateline_labels))
        labels=dateline_labels

    e_values = []
    s_values = []
    h_scores = []
    c_scores = []
    if not SKIP_ELBOW:
        for e in np.arange(MIN_EPS, MAX_EPS, STEP_EPS):
            for s in range(MIN_SAMP, MAX_SAMP, STEP_SAMP):
                h_score, c_score, dbscan, svd_data = cluster(combined_svd_vector, labels, s, e)
                e_values.append(e)
                s_values.append(s)
                h_scores.append(h_score)
                c_scores.append(c_score)

    # Plot h scores vs k
    fig = plt.figure(1)
    ax=fig.add_subplot(111,projection='3d')
    ax.set_title('DBSCAN Homogeneity vs Min-Samples & Epislon Radius \nFeature Vectors and Places Labels')
    ax.set_xlabel('Epsilon Radius')
    ax.set_ylabel('Min-Samples')
    ax.set_zlabel('Homogeneity')
    ax.scatter(e_values, s_values, h_scores, s=2)
    plt.show()

    min_samples = int(input('min-samples to use for plot:'))
    eps = float(input('eps to use for plot:'))
    plot_clustering(combined_svd_vector, labels, min_samples, eps)

main()

