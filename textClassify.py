# coding: utf-8

import os, os.path
import jieba
import csv
import sklearn.feature_extraction
import sklearn.naive_bayes as nb
import sklearn.externals.joblib as jl
import sys

from extract import *
import cluster as cl

TRAIN_DATA_PATH = 'data'
TRAIN_TEXT_FILE = 'train_file.txt'

train_data = {'Economy':[], 'Technology':[], 'Culture':[],  'Education':[], 'Sports':[], 'Health':[], 'Politics':[], 'Travel':[]}
target_map = {'Culture':0, 'Economy':1, 'Education':2, 'Health':3, 'Politics': 4, 'Sports':5, 'Technology':6, 'Travel':7}

def load_train_data(parent_dir):
    for parent, dirnames, filenames in os.walk(parent_dir):
        for dirname in dirnames:
            for par, dirs, fnames in os.walk(os.path.join(parent, dirname)):
                for fname in fnames[:500]:
                    file = open(os.path.join(par, fname))
                    for line in file:
                        train_data[dirname].append(line)
                    file.close()
def clipper(txt):
    return jieba.cut(txt)

def train_classifier(train_data):
    gnb = nb.MultinomialNB(alpha = 0.01)
    fh = sklearn.feature_extraction.FeatureHasher(n_features=15000,non_negative=True,input_type='string')
    
    
    kvlist, targetlist = [], []
    for key in train_data:
        for value in train_data[key]:
            kvlist += [[i for i in clipper(value)]]
            targetlist += [target_map[key]]

    #print kvlist[:5], targetlist[:5]
    X = fh.fit_transform(kvlist)
    gnb.fit(X, targetlist)
    return fh, gnb


def load_test_data():
    file = open(os.path.join(DATA_PATH, TEST_FILE))
    test_data = []
    test_target = []
    for line in file:
        values = line.split('\t')
        test_data.append(values[1])
        test_target.append(values[0])
    return test_data, test_target

def predict_cascade_topics(cascades, fh, gnb):
    file = open('data\\cascade_topics.txt', 'w')
    
    for cascade in cascades:
        text = cascade.root.text
        kv = [t for t in clipper(text)]
        mt = fh.transform([kv])
        num = gnb.predict(mt)
        for key in target_map:
            if target_map[key] == num:
                file.write(cascade.mid+'\t'+key+'\n')
    file.close()

if __name__ == "__main__":
    n_cluster = 5
    load_train_data()

    users = load_users(os.path.join(DATA_PATH, PROFILE_FILE))
    print 'users:', len(users)
    cascades = load_cascades(os.path.join(DATA_PATH, PROCESS_FILE), users, count =33214, threshold=10)
    print 'cascades:', len(cascades)
    
    fh, gnb = train_classifier(train_data)
    predict_cascade_topics(cascades, fh, gnb)

    
