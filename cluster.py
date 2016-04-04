import networkx as nx
import numpy as np
from sklearn import cluster, metrics, datasets
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import pairwise_distances
from sklearn.utils.graph import graph_laplacian
from sklearn.manifold.spectral_embedding_ import _set_diag
from sklearn.utils.extmath import _deterministic_vector_sign_flip
from sklearn.metrics import pairwise_distances, euclidean_distances
from sklearn.utils.arpack import eigsh
from sklearn.cluster import k_means
from scipy.sparse import coo_matrix
import math, os.path, random
import heapq

from extract import *
from kernel import *

import cascade as cs
from wtime import WTime


def construct_sparse_matrix(cascades, measure='weighted', beta=5, threshold=30, n_neighbors = 6, gamma=5.0):
    row, col, dis_data, sim_data = [], [], [], []
    vectors, mu = [], []
    if measure == 'MDS':
        seed = np.random.RandomState(seed=3)
        mds = manifold.MDS(n_components=1, max_iter=3000, eps=1e-9, random_state=seed, dissimilarity="precomputed", n_jobs=1)
        for cascade in cascades:
            vectors.append(MDS(cascade, mds, n=30))
    elif measure == 'GMDS':
        for cascade in cascades:
            vectors.append(GMDS(cascade, n=30))
    elif measure == 'weighted':
        for cascade in cascades:
            vectors.append(weighted_degree_vector(cascade, threshold))
    elif measure == 'tfidf':
        for cascade in cascades:
            vectors.append(tfidf(cascade, k=threshold, gamma=gamma))
    elif measure == 'PCA' or measure == 'PCA_cos':
        for cascade in cascades:
            v, m = PCA_vector(cascade, threshold)
            vectors.append(v)
            mu.append(m)
    elif measure == 'entropy':
        for cascade in cascades:
            vectors.append(entropy(cascade, threshold))
    elif measure == 'wiener':
        for cascade in cascades:
            vectors.append(wiener_index(cascade, threshold))

    print 'generating matrix'
    for i in range(len(cascades)):
        dis_neighbors = {}
        for j in range(i, len(cascades)):
            if i != j:
                C1, C2 = cascades[i], cascades[j]
                if measure == 'MDS':
                    dis = distance(vectors[i][:5], vectors[j][:5], n=threshold, type='eu')
                elif measure == 'GMDS':
                    dis = distance(vectors[i][:30], vectors[j][:30], n=threshold, type='eu')
                elif measure == 'weighted' or measure == 'tfidf':
                    dis = distance(vectors[i], vectors[j], n=threshold, type='eu')
                elif measure == 'PCA':
                    dis = distance(vectors[i], vectors[j], n=min(len(vectors[i]), len(vectors[j])), type='eu')
                elif measure == 'random':
                    dis = random_walk_kernel(C1, C2, n=threshold)
                elif measure == 'entropy' or measure == 'wiener':
                    dis = abs(vectors[i]-vectors[j])
                dis_neighbors[j] = dis
        #print i, dis_neighbors
        dis_neighbors = sorted(dis_neighbors.items(), lambda x, y: cmp(x[1], y[1]), reverse = False)
        for item in dis_neighbors[:n_neighbors]:
            row.append(i)
            col.append(item[0])
            dis_data.append(item[1])
        #print i, row, col, dis_data
        if i%100==0:
            print i, len(row)

    std = np.std(np.array(dis_data))
    sim_data = np.exp(-beta*np.array(dis_data)/std)
    print len(row), gamma, measure

    dis_A = coo_matrix((dis_data, (row, col)), shape=(len(cascades), len(cascades)))
    sim_A = coo_matrix((sim_data, (row, col)), shape=(len(cascades), len(cascades)))
    #dis_A = []
    return dis_A, sim_A

def save_cluster_labels(cascades, y, labels, filename):
    try:
        file = open(filename, 'w')
        for i, cascade in enumerate(cascades):
            mid = cascade.mid
            line = [mid, str(labels[i])]
            line.extend([str(t) for t in y[i]])
            file.write('\t'.join(line) +'\n')
    finally:
        file.close()

def load_cluster_labels(filename):
    print filename
    cluster = {}
    try:
        file = open(filename)
        for line in file:
            values = line.strip('\n').split('\t')
            cluster[values[0]] = int(values[1])
    finally:
        file.close()
    return cluster
        
def spcluster(A, n_cluster, n_neighbors):
    SC = cluster.SpectralClustering(n_clusters=n_cluster, affinity='precomputed', n_neighbors=n_neighbors, eigen_solver='arpack')
    
    labels = SC.fit_predict(A)
    print silhouette_score(A, labels)
    return labels

def kmeanscluster(A, n_cluster):
    kmeans = cluster.KMeans(n_clusters=n_cluster).fit(A)

    print 'n_cluster', n_cluster, len(kmeans.labels_)
    labels, centers = kmeans.labels_, kmeans.cluster_centers_
    #print labels, n_cluster, silhouette_score(A, labels)
    print 'len of labels', len(labels), type(A)
    return labels

def agglomerativecluster(A, n_cluster):
    agg = cluster.AgglomerativeClustering(n_clusters=n_cluster).fit(A)
    labels = agg.labels_
    return labels
                   
def spectralcluster(A, n_cluster, n_neighbors=6, random_state=None, eigen_tol=0.0):
    #maps = spectral_embedding(affinity, n_components=n_components,eigen_solver=eigen_solver,random_state=random_state,eigen_tol=eigen_tol, drop_first=False)

    # dd is diag
    laplacian, dd = graph_laplacian(A, normed=True, return_diag=True)
    # set the diagonal of the laplacian matrix and convert it to a sparse format well suited for e    # igenvalue decomposition
    laplacian = _set_diag(laplacian, 1)
    
    # diffusion_map is eigenvectors
    # LM largest eigenvalues
    laplacian *= -1
    eigenvalues, eigenvectors = eigsh(laplacian, k=n_cluster,
                                   sigma=1.0, which='LM',
                                   tol=eigen_tol)
    y = eigenvectors.T[n_cluster::-1] * dd
    y = _deterministic_vector_sign_flip(y)[:n_cluster].T

    random_state = check_random_state(random_state)
    centroids, labels, _ = k_means(y, n_cluster, random_state=random_state)

    return eigenvalues, y, centroids, labels

def get_cluster_label(C, cascades, eigenvalues, eigenvectors, centroids, labels, k = 5):
    n = len(cascades)
    v = tfidf(C, k = k)
    L = []
    vectors = []
    for i, cascade in enumerate(cascades):
        vectors.append(tfidf(cascade, k=k))
        L.append(distance(v, vectors[i], n=k, type='eu'))
    L.append(sum(distance))

    y = [0.0 for i in range(n)]
    for i in range(k):
        for j in range(n):
            y[i] += L[j]*eigenvectors[i][j]
        y[i] /= float(eigenvalues[i])
    
    min = sys.maxint
    index = 0
    for i, centrod in enumerate(centroids):
        dis = euclidean_distances(y, centroid)
        if dis < min:
            min = dis
            index = i
    return i   

if __name__ == "__main__":
    threshold = 30
    count = 33214
    method = 'tfidf'
    gamma = 5.0
    alg = 'sp'
    n_cluster=5

    users = load_users(os.path.join(DATA_PATH, PROFILE_FILE))
    print 'users:', len(users), PROCESS_FILE
    cascades = load_cascades(os.path.join(DATA_PATH, PROCESS_FILE), users, count =count, threshold=threshold)
    print 'cascades:', len(cascades)

    n_neighbors = n_cluster*4
    dis_A, sim_A = construct_sparse_matrix(cascades, measure=method, threshold = threshold, n_neighbors=n_neighbors, gamma=gamma)
    print 'begin spectral clustering'
    
    if alg == "sp":
        y = [[0 for i in range(n_cluster)] for j in range(count)]
        labels = spcluster(sim_A, n_cluster, n_neighbors)   # use similarity
    elif alg == "kmeans":
        labels = kmeanscluster(dis_A, n_cluster)  # use dis, euclidean
    elif alg == "agg":
        labels = agglomerativecluster(dis_A, n_cluster)

    print 'clustered'
    numbers = {i:0 for i in range(n_cluster)}
    print len(labels), len(numbers)
    for i in range(len(labels)):
        numbers[labels[i]]+=1
    print numbers
    save_cluster_labels(cascades, y, labels, 'data\\'+str(count)+'_'+method+'_cluster_'+alg+'_'+str(threshold)+'_'+str(gamma))

    

    
