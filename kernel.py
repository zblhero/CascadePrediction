import networkx as nx
import numpy as np
from numpy import *
from sklearn import cluster, metrics, datasets
from sklearn.metrics import pairwise_distances, euclidean_distances
from sklearn.isotonic import IsotonicRegression
from sklearn.decomposition import PCA
from sklearn.utils import check_random_state
from scipy.sparse import coo_matrix
from scipy.spatial.distance import minkowski
import math, os.path, random
from sklearn import manifold

#from extract import *

import cascade as cs
from wtime import WTime

def euclidean(v1, v2, n):
    k = 0.0
    for i in range(min(len(v1), len(v2), n)):
        k += (v1[i]-v2[i])*(v1[i]-v2[i])
                   
    return math.sqrt(k)

def cosine_similarity(v1,v2, mu1, mu2):
    "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]; y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    return (sumxy/math.sqrt(sumxx*sumyy))*exp(-100*abs(mu1-mu2))
    

def kl_divergence(v1, v2, n=30):
    k = 0.0
    for i in range(min(len(v1), len(v2), n)):
        v1[i] = abs(v1[i])
        v2[i] = abs(v2[i])
        if v1[i] != 0:
            k += abs(v1[i]*math.log(2*v1[i]/float(v1[i]+v2[i])))
        if v2[i] != 0:
            k += abs(v2[i]*math.log(2*v2[i]/float(v1[i]+v2[i])))
    return k

def distance(v1, v2, n=30 , type='eu'):
    if type == 'eu':
        return euclidean(v1, v2, n)
    elif type == 'kl':
        return kl_divergence(v1, v2, n)

def kronecker_product(A, B):
    n = len(A)
    m = len(A[0])
    p = len(B)
    q = len(B[0])
    C = [[0 for i in range(m*q)] for j in range(n*p)]
                   
    for i in range(n):
        for x in range(m):
            for j in range(p):
                for y in range(q):
                    C[i*p+j][x*q+y] = A[i][x]*B[j][y]
    return C

def get_matrix(C, n, alpha=1, sigma = 1, type='sparse'):
    n = min(n, len(C.nodes))
    if type == 'sparse':
        row, col, data = [], [], []
    else:
        M = [[0 for i in range(n)] for j in range(n)]
    
    
    for i in range(n):
        for j in range(i, n):
            if C.nodes[j].parent == C.nodes[i]:
                delta_t = abs(C.nodes[j].created_at.mt-C.nodes[i].created_at.mt)/1000
                if type == 'sparse':
                    row.append(i)
                    col.append(j)
                    data.append(alpha*math.exp(-delta_t*alpha))
                else:
                    M[i][j] = alpha*math.exp(-delta_t*alpha)
    if type == "sparse":
        M = coo_matrix((data, (row, col)), shape=(n, n))
    return M

def weighted_degree_vector(C, n=30):
    sigma, virality = 0.0, 0.0
    v = [0 for i in range(n)]
    
    for i in range(min(len(C.nodes), n)):
        node = C.nodes[i]
        for child in node.children:
            delta_t = abs(child.created_at.mt-node.created_at.mt)/1000
            weight = math.exp(-delta_t)
            sigma += weight
            if i<30:
                virality += weight
            v[i] += weight
    v = sorted(v, reverse=True)
    return v

def weighted_degree_kernel(C1, C2, n, measure='kl'):
    '''extract the first 20 nodes of each cascade'''
    sigma1, sigma2 = 0, 0
    v1, v2 = [0 for i in range(n)], [0 for i in range(n)]
    
    for i in range(n):
        node = C1.nodes[i]
        for child in node.children:
            delta_t = abs(child.created_at.mt-node.created_at.mt)/1000
            weight = math.exp(-delta_t)
            sigma1 += weight
            v1[i] += weight

        node = C2.nodes[i]
        for child in node.children:
            delta_t = abs(child.created_at.mt-node.created_at.mt)/1000
            weight = math.exp(-delta_t)
            sigma2 += weight
            v2[i] += weight

    v1 = array(sorted(v1, reverse = True))
    v2 = array(sorted(v2, reverse = True))
                   
    k = 0.0
    if measure == 'euclidean':
        k = euclidean(v1, v2, n)
    elif measure == 'kl':
        k = abs(kl_divergence(v1, v2, n))
    #print 'Weighted: %.3f %.3f'%(k, euclidean(v1, v2, n)), v1[:2], v2[:2]
    return abs(k)

def random_walk_kernel(C1, C2, n=30):
    n = min(n, len(C1.nodes), len(C2.nodes))
    p1 = [1/float(n) for i in range(n)]
    q1 = [1/float(n) for i in range(n)]

    A1 = [[0.0 for i in range(n)] for j in range(n)] 
    A2 = [[0.0 for i in range(n)] for j in range(n)]
                   
    for i in range(n):
        for j in range(n):
            if C1.nodes[i] in C1.nodes[j].children or C1.nodes[j] in C1.nodes[i].children:
                A1[i][j] = 1
            if C2.nodes[i] in C2.nodes[j].children or C2.nodes[j] in C2.nodes[i].children:
                A2[i][j] = 1

    W = mat(kronecker_product(A1, A2))
    
    lamda = 0.5
    Ak = eye(n*n)-lamda*W  
  
    k = Ak.I.sum().sum()
    #print 'Random Walk:', abs(k)
    return abs(k)

def entropy(C, n, show=False):
    entropy, degree = [], {}
    n = min(len(C.nodes), n)
    e = 0.0
    root = C.nodes[0]
    for i in range(1, n):
        node = C.nodes[i]
        parent = node.parent

        delta_t = abs(node.created_at.mt-root.created_at.mt)/1000
        weight = math.exp(-delta_t)
        if degree.has_key(parent):
            degree[parent] += 1
        else:
            degree[parent] = 1

        probability = float(degree[parent])/(i+1)
        if show:
            print probability, weight, -probability*log(weight*probability)
        e += -probability*log(weight*probability)
        entropy.append(e)

    return e/n
        


def entropy_kernel(C1, C2, n):
    e1, e2 = 0, 0
    d1, d2 = {}, {}
    for i in range(1,n):
        n1, n2 = C1.nodes[i], C2.nodes[i]
        p1, p2 = n1.parent, n2.parent
        
        delta_t1 = abs(n1.created_at.mt-p1.created_at.mt)/1000
        weight1 = math.exp(-delta_t1)
        delta_t2 = abs(n2.created_at.mt-p2.created_at.mt)/1000
        weight2 = math.exp(-delta_t2)

        if d1.has_key(p1):
            d1[p1] += 1
        else:
            d1[p1] = 1
        if d2.has_key(p2):
            d2[p2] += 1
        else:
            d2[p2] = 1

        pro1 = float(d1[p1])/(i+1)
        e1 += -pro1*log(weight1*pro1)
        
        pro2 = float(d2[p2])/(i+1)
        e2 += -pro2*log(weight2*pro2)
        #print i, d1[p1], d2[p2], pro1, pro2
    #print 'Entropy: ', abs(e1-e2), e1, e2
    return abs(e1-e2)

def wiener_index(C, k=30):
    w = 0.0
    for vi in C.nodes[:k]:
        w += vi.depth
    return w/k

def wiener_index_kernel(C1, C2, n=30):
    if n==-1:
        n1 = len(C1.nodes)
        n2 = len(C2.nodes)
    else:
        n1, n2 = n, n
    
    k1 = 0.0
    for vi in C1.nodes[:n1]:
        for vj in C1.nodes[:n1]:
            try:
                k1 += d[vi.uid][vj.uid]
            except KeyError:
                pass
    k1 /= float(n1*(n1-1))
    
    k2 = 0.0
    for vk in C2.nodes:
        for vl in C2.nodes:
            try:
                k2 += d[vk.uid][vl.uid]
            except KeyError:
                pass
    k2 /= float(n2*(n2-1))
    print 'Weiner: ', abs(k1-k2), k1, k2
    return abs(k1-k2)

def PCA_vector(C, k=30, nc=1):
    M = get_matrix(C, k, type='dense')
    pca = PCA(n_components = nc)

    evc = [l[0] for l in pca.fit_transform(M)]
    return evc[:k]

def PCA_kernel(C1, C2, n, nc=1, measure='kl'):
    M1 = get_matrix(C1, n, type='normal')
    M2 = get_matrix(C2, n, type='normal')

    pca = PCA(n_components = nc)

    evc1 = [l[0] for l in pca.fit_transform(M1)]
    evc2 = [l[0] for l in pca.fit_transform(M2)]

    k = 0.0
    if measure == 'euclidean':
        k = euclidean(evc1, evc2, n)
    elif measure == 'kl':
        k = kl_divergence(evc1, evc2, n)
    #print 'PCA: %.3f %.3f'%(k, euclidean(evc1, evc2, n)), evc1[:2], evc2[:2]
    return abs(k)

def GMDS(M, pairwise_distance, n=30, max_iter=300, eps=1e-3):
    delta = np.array([[0.0 for i in range(len(M))] for j in range(len(M))])
    for i in range(len(M)):
        for j in range(len(M)):
            distance = 0.0
            for k in range(len(M)):
                distance += (M[i][k]-M[j][k])
            delta[i][j] = distance
            
    #metric=true, dissimilarity=precomputed
    n_samples = len(M)
    n_components = 1
    random_state = check_random_state(np.random.RandomState(seed=3))
    #random_state = check_random_state(5)
    X = random_state.rand(n_samples * n_components)
    X = X.reshape((n_samples, n_components))

    old_stress = None
    for it in range(max_iter):
        dis = np.array([[(X.ravel()[j]-X.ravel()[i]) for i in range(X.shape[0])]for j in range(X.shape[0])])
        disparities = delta/pairwise_distance
        print 'delta', delta
        print 'disparities', disparities

        #dis = np.array([[(X.ravel()[j]-X.ravel()[i]) for i in range(X.shape[0])]for j in range(X.shape[0])])
        #disparities = delta

        stress = (((dis.ravel() - disparities.ravel())) ** 2).sum() / 2

        # Update X using the Guttman transform
        dis[dis == 0] = 1e-5
        ratio = disparities / dis
        B = - ratio
        B[np.arange(len(B)), np.arange(len(B))] += ratio.sum(axis=1)
        X = 1. / n_samples * np.dot(B, X)
        X = X - min(X.ravel())

        dis = np.sqrt((X ** 2).sum(axis=1)).sum()
        if old_stress is not None:
            if(old_stress - stress / dis) < eps:
                #print 'it, stress', it, stress
                break
        old_stress = stress / dis

        
    #matrix = mds.fit(array(dis)).embedding_
    print X.ravel()
    vector = sorted(X.ravel(), reverse=True)
    #print vector
    return vector     

def tfidf(C, k=30, gamma=5.0):
    sigma, virality = 0.0, 0.0
    v = [0 for i in range(k)]
    
    for i in range(min(len(C.nodes), k)):
        node = C.nodes[i]
        for child in node.children:
            delta_t = abs(child.created_at.mt-node.created_at.mt)/1000
            weight = math.exp(-delta_t)
            v[i] += weight
        #print node.name, node.depth, v[i]
        if gamma != -1:
            v[i] *= float(gamma+node.depth)
    v = sorted(v, reverse=True)
    return v
    
