import networkx as nx
from numpy import *
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib import cm
from spectral import *  # dimension reduction
from sklearn import cluster, metrics, datasets
from sklearn.metrics import pairwise_distances

import scipy as sp
import scipy.stats
from scipy.stats.stats import pearsonr

import math, os.path, codecs
from datetime import datetime

from extract import *
from kernel import *
from textClassify import *
from wtime import WTime
import cascade as cs
import cluster as cl

root_dir = 'cascades'
cascades = []
k = 6
interval = 600
d= {}

def total_volumn(labels):
    total = [0 for i in range(k)]
    for i in range(len(cascades)):
        total[labels[cascades[i].mid]] += cascades[i].forward
    return total

def get_max_degree(cascade):
    max = 0.0
    total = 0.0
    for node in cascade.nodes:
        if len(node.children) > max:
            max = len(node.children)
        total += len(node.children)
    return max

def max_degrees(labels):
    max = [0.0 for i in range(k)]
    cluster_num = [0.0 for i in range(k)]
    for i in range(len(cascades)):
        max[labels[cascades[i].mid]] += get_max_degree(cascades[i])
        cluster_num[labels[cascades[i].mid]] += 1
    for i in range(k):
        max[i] /= cluster_num[i]
    return max

def average_size(labels):
    size = [0.0 for i in range(k)]
    cluster_num = [0 for i in range(k)]
    for i in range(len(cascades)):
        size[labels[cascades[i].mid]] += cascades[i].forward
        cluster_num[labels[cascades[i].mid]] += 1
    for i in range(k):
        size[i] /= cluster_num[i]
    return size, cluster_num

def average_depth(labels):
    depth = [0.0 for i in range(k)]
    cluster_num = [0 for i in range(k)]
    for i in range(len(cascades)):
        depth[labels[cascades[i].mid]] += cascades[i].depth
        cluster_num[labels[cascades[i].mid]] += 1
    for i in range(k):
        depth[i] /= cluster_num[i]
    return depth
        
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, m-h, m+h

def correlation(cascades, k):
    stats = {}
    stat_file = open(os.path.join(DATA_PATH, STAT_FILE))
    for line in stat_file:
        values = line.split('\t')
        stats[values[0]] = [int(values[1]), int(values[2]), int(values[3]), float(values[4])]


    centralities = [[] for i in range(10)]
    size = []
    depth = []
    wiener = []
    for cascade in cascades:
        mid = cascade.mid
        vector = tfidf(cascade, k=k, gamma=5.0)
        for i in range(10):
            centralities[i].append(vector[i])
        size.append(stats[mid][0])
        wiener.append(stats[mid][1])
        depth.append(stats[mid][2])
        #print len(centralities), len(centralities[0]), len(size)
    print k
    for i in range(10):
        print i, pearsonr(centralities[i], size)[0], pearsonr(centralities[i], wiener)[0], pearsonr(centralities[i], depth)[0]
    print len(cascades), len(cascades[0].nodes), tfidf(cascades[0], k=k, gamma=5.0)
        

def stats(stat_filename, cluster_filename, n_cluster):
    stat_file = open(stat_filename)
    cluster_file = open(cluster_filename)
    stats = {}
    
    for line in cluster_file:
        values = line.split('\t')
        stats[values[0]]=[int(values[1])]
    for line in stat_file:
        values = line.split('\t')
        if stats.has_key(values[0]):
            stats[values[0]].extend([int(values[1]), int(values[2]), int(values[3]), float(values[4])])
    
            
    print 'stats:', len(stats)
    average_size = {i:0.0 for i in range(n_cluster)}
    average_depth = {i:0.0 for i in range(n_cluster)}
    cluster_num = {i:0.0 for i in range(n_cluster)}
    average_max_degree = {i:0.0 for i in range(n_cluster)}
    average_wiener = {i:0.0 for i in range(n_cluster)}
    size = {i:[] for i in range(n_cluster)}
    depth = {i:[] for i in range(n_cluster)}
    for mid in stats:
        average_size[stats[mid][0]] += stats[mid][1]
        average_depth[stats[mid][0]] += stats[mid][2]
        cluster_num[stats[mid][0]] += 1.0
        average_max_degree[stats[mid][0]] += stats[mid][3]
        average_wiener[stats[mid][0]] += stats[mid][4]
        size[stats[mid][0]].append(stats[mid][1])
        depth[stats[mid][0]].append(stats[mid][2])

    
    for i in range(n_cluster):
        average_size[i] /= cluster_num[i]
        average_depth[i] /= cluster_num[i]
        average_max_degree[i] /= cluster_num[i]
        average_wiener[i] /= cluster_num[i]

    print cluster_num, average_size, average_depth, average_max_degree, average_wiener
    return size, depth
