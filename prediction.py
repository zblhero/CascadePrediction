#coding:utf-8
import networkx as nx
from numpy import *
from spectral import *  # dimension reduction
from sklearn.metrics import *
from sklearn import cluster, metrics, datasets, cross_validation, svm
from sklearn.metrics import pairwise_distances
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator
import cascade as cs
import cluster as cl
import math, os.path, copy
from datetime import datetime
import random, numpy

from extract import *
from kernel import *
from textClassify import load_topics

import sys
reload(sys)
sys.setdefaultencoding('utf-8')


graph = nx.DiGraph()
topics = {}

class DistanceClassifier(BaseEstimator):
    def __init__(self):
        self.distance = {}

    def fit(self, x_train, y_train):
        for i in range(len(x_train)):
            x, y = x_train[i], y_train[i]
            if not self.distance.has_key(x):
                self.distance[x] = 0
            if y == 1:
                self.distance[x] += 1
            elif y == 0:
                self.distance[x] -= 1
        return self
    

    def predict(self, x_target):
        result = []
        for x in x_target:
            if self.distance[x] >= 0:
                result.append(1)
            elif self.distance[x] <0 :
                result.append(0)
        return np.array(result)

    def score(self, x_target, y_target):
        result = []
        for x in x_target:
            if self.distance[x] >= 0:
                result.append(1)
            elif self.distance[x] <0 :
                result.append(0)
        num = 0
        for i in range(len(result)):
            if y_target[i] == result[i]:
                num += 1
        return float(num)/len(result)

class RandomGuess(BaseEstimator):
    def fit(self, x_train, y_train):
        self.forward = x_train
        return self
    def predict(self, x_target):
        result = []
        for x in x_target:
            if random.random() <0.5:
                result.append(1)
            else:
                result.append(0)
        return np.array(result)

    def score(self, x_target, y_target):
        result = self.predict(x_target)
        num = 0
        for i in range(len(result)):
            if y_target[i] == result[i]:
                num += 1
        return float(num)/len(result)



def extract_root_feature(cascade):
    root = cascade.root
    if root.gender == 'ÄÐ':
        gender = 1
    else:
        gender = 0
    return [root.follower_num/10000000.0, gender, root.weibo_num/10000.0, root.verification]

def extract_temporal_feature(cascade, k=30):
    feature = []
    first = 0
    last = 0
    for i in range(1, min(len(cascade.nodes), k)):
        time = (cascade.nodes[i].created_at.mt - cascade.root.created_at.mt)/60.0
        if i <= k/2:
            first += (cascade.nodes[i].created_at.mt - cascade.nodes[i-1].created_at.mt)/60.0
        else:
            last += (cascade.nodes[i].created_at.mt -cascade.nodes[i-1].created_at.mt)/60.0
    for i in range(10):
        time = (cascade.nodes[i].created_at.mt - cascade.root.created_at.mt)/60.0
        #print len(cascade.nodes), i
        feature.append(time)
    feature.append(first*2/k)
    feature.append(last*2/k)
    return feature

def extract_structural_feature(cascade, k=30):
    feature = []
    orig_connections = 0
    #print len(graph.nodes()), len(graph.edges()), cascade.root.name
    border_nodes = set(graph.neighbors(cascade.root.name))
    depth = 0
    for i in range(0, min(len(cascade.nodes), k)):
        degree = graph.degree(cascade.nodes[i].name)
        if cascade.nodes[i] in graph.neighbors(cascade.root.name) and cascade.root in graph.neighbors(cascade.nodes[i].name):
            orig_connections += 1
        border_nodes = border_nodes | set(graph.neighbors(cascade.nodes[i].name))
        depth += cascade.nodes[i].depth
    for i in range(10):
        degree = graph.degree(cascade.nodes[i].name)
        feature.append(degree)
    border_nodes = border_nodes - set(cascade.nodes)
    feature.append(len(border_nodes)/10000.0)
    feature.append(float(depth)/k)
    #print len(border_nodes), depth, k, 
    return feature

def extract_content_feature(cascade):
    text = cascade.root.text
    link, subject, mention = 0,0,0
    if text.find('http://t.cn') >= 0:
        link = 1
    if text.find('#') >= 0:
        subject = 1
    if text.find('@') >= 0:
        mention = 1

    content_feature = [0 for i in range(8)]
    for i in range(9):
        if topics[cascade.mid] == i:
            content_feature[i] = 1
    content_feature.extend([link, subject, mention])
    #print 'content_feature', cascade.mid, content_feature
    return content_feature


def extract_feature(cascade, feature_type, k=30):
    feature = []
    user = cascade.root
    
    root_feature = extract_root_feature(cascade)
    temporal_feature = extract_temporal_feature(cascade, k=k)
    structural_feature = extract_structural_feature(cascade, k=k)
    content_feature = extract_content_feature(cascade)
    
    feature.extend(root_feature)
    feature.extend(temporal_feature)
    feature.extend(structural_feature)
    feature.extend(content_feature)
    
    embed_feature = tfidf(cascade, k=k, gamma=5.0)[:5]
    
    if feature_type == 'embedded':
        feature.extend(embed_feature)
    if feature_type == "temporal":
        return temporal_feature
    if feature_type == "embedded_only":
        return embed_feature
    if feature_type == "no_embedded":
        return feature
    if feature_type == "none":
        return []
    if feature_type == 'PCA':
        PCA_feature = PCA_vector(cascade, k=k)[:10]
        feature.extend(PCA_feature)
    if feature_type == 'wiener':
        wiener_feature = wiener_index(cascade, k=k)
        feature.append(wiener_feature)
    return feature

def predict_structure(filename, wiener, model='logisitic', feature_type='embedded', w_tau = 2, k=30):
    data, target = [], []
    for i, cascade in enumerate(cascades):
        feature = extract_feature(cascade, feature_type, k=k)
        data.append(feature)
        
        t = int(wiener[cascade.mid] > w_tau)
        target.append(t)

    print feature_type, data[0], target[0]
    wide = 0
    for i, t in enumerate(target):
        if t == 1:
            wide+=1
    print len(data), wide, len(data[0])
    if model == 'logistic':
        clf = LogisticRegression(C=1.0)
        clf.fit(data, target)
        #print ' '.join(["%.3f"%x for x in clf.coef_[0]])
    elif model == 'svm':
        clf = svm.SVC(kernel = 'linear', C=1)
    elif model == 'randomforest':
        clf = RandomForestClassifier(max_depth=5, max_features=1)
        clf.fit(data, target)
    elif model == 'distance':
        data = labels
        clf = DistanceClassifier()
    elif model == "random":
        clf = RandomGuess()
    
    accuracy = cross_validation.cross_val_score(clf, data, target, cv=5)
    precision = cross_validation.cross_val_score(clf, data, target, cv=5, scoring='precision')
    recall = cross_validation.cross_val_score(clf, data, target, cv=5, scoring='recall')
    #scores = cross_validation.cross_val_score(clf, data, target, cv=5)
    F1 = cross_validation.cross_val_score(clf, data, target, cv=5, scoring='f1')
    print model, feature_type
    print  np.average(precision), np.average(recall), np.average(F1), np.average(accuracy)

def predict_virality(cascades, model='randomforest', feature_type='embedded', predict_target='viral', tau = 100, k=30):
    #data, target = load_train_data(filename, cascades, labels, feature_type, predict_target='viral')

    data, target = [], []
    for cascade in cascades:
        feature = extract_feature(cascade, feature_type, k=k)
        data.append(feature)
        
        if predict_target == 'viral':
            t = int(cascade.forward > tau)
            target.append(t)

    print feature_type, data[0], target[0], len(data[0])

    #print data[0], target[0]
    if model == 'logistic':
        clf = LogisticRegression(C=1.0)
        #print len(data[1]), data[1], target[1]
        clf.fit(data, target)
        print ' '.join(["%.3f"%x for x in clf.coef_[0]])
    elif model == 'svm':
        clf = svm.SVC(kernel = 'linear', C=1)
    elif model == 'randomforest':
        clf = RandomForestClassifier(max_depth=5, max_features=1)
    elif model == 'distance':
        data = labels
        clf = DistanceClassifier()
    elif model == "random":
        #data = size
        clf = RandomGuess()

    accuracy = cross_validation.cross_val_score(clf, data, target, cv=5)
    precision = cross_validation.cross_val_score(clf, data, target, cv=5, scoring='precision')
    recall = cross_validation.cross_val_score(clf, data, target, cv=5, scoring='recall')
    F1 = cross_validation.cross_val_score(clf, data, target, cv=5, scoring='f1')
    #scores = cross_validation.cross_val_score(clf, data, target, cv=5)
    print model, feature_type, len(data[0])
    print  np.average(precision), np.average(recall), np.average(F1), np.average(accuracy)



if __name__ == "__main__":
    k=30
    count = 33214
    method = 'tfidf'
    alg = 'sp'
    n_cluster = 6
    gamma = 5
    model = 'logistic'

    users = cl.load_users(os.path.join(cl.DATA_PATH, cl.PROFILE_FILE))
    print 'users:', len(users)
    cascades = cl.load_cascades(os.path.join(cl.DATA_PATH, cl.PROCESS_FILE), users, count = count, threshold=k)
    print 'cascades:', len(cascades)
    graph = cl.load_graph(cascades)
    print 'graph:', len(graph.nodes()), len(graph.edges())
    topics = load_topics(os.path.join(cl.DATA_PATH, cl.TOPIC_FILE))

    tau = 100
    predict_virality(cascades, tau=tau, model=model, feature_type='embedded', predict_target='viral', k=k)
    predict_virality(cascades, tau=tau, model=model, feature_type='no_embedded', predict_target='viral', k=k)
    predict_virality(cascades, tau=tau, model=model, feature_type='PCA', predict_target='viral', k=k)
    predict_virality(cascades, tau=tau, model=model, feature_type='wiener', predict_target='viral', k=k)
    predict_virality(cascades, tau=tau, model='random', predict_target='viral', k=k)


    # predict structure
    '''w_tau = 2.05
    stat_file = open(os.path.join(DATA_PATH, STAT_FILE))
    wiener = {}
    for line in stat_file:
        values = line.split('\t')
        wiener[values[0]] = float(values[4])
    print k, count, w_tau, model, len(wiener)
    predict_structure(cascades, wiener, model=model, feature_type='embedded', k=k, w_tau=w_tau)
    predict_structure(cascades, wiener, model=model, feature_type='no_embedded', k=k, w_tau=w_tau)
    predict_structure(cascades, wiener, model=model, feature_type='PCA', k=k, w_tau=w_tau)
    predict_structure(cascades, wiener, model=model, feature_type='wiener', k=k, w_tau=w_tau)
    predict_structure(cascades, wiener, model='random', k=k, w_tau=w_tau)'''

