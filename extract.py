# coding:utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import urllib, urllib2
import re, time, socket, codecs
from bs4 import BeautifulSoup
import os, os.path
import copy
import networkx as nx

import cascade as cs
from wtime import WTime
#from cluster import load_cluster_labels

WEIBO = 'http://weibo.cn'
DATA_PATH = 'E:\\Data Set\\Weibo'
CASCADE_FILE = 'cascade_file.txt'
USER_FILE = 'user_list.txt'
FORWARD_FILE = 'forward_file.txt'
PROCESS_FILE = 'process_file.txt'
PROFILE_FILE = 'profile_file.txt'
STAT_FILE = 'stat_file.txt'
TOPIC_FILE = 'cascade_topics.txt'

graph = nx.DiGraph()
def load_graph(cascades):
    graph = nx.DiGraph()
    for cascade in cascades:
        for node in cascade.nodes:
            graph.add_node(node.name)
            for child in node.children:
                graph.add_edge(node.name, child.name)
    return graph

def load_cascades(filename, users, type='tree', count = 200000, threshold = 10000000, DEBUG=False):
    cascades = []
    file = open(filename)
    read, total, line_num, cascade_num = 0, 0, 0, 0
    while True:
        if read == 1:
            line = forward_line
            read = 0
        else:
            line = file.readline()
            line_num += 1
        if line.startswith(codecs.BOM_UTF8):
            line = line[3:] 
        if line == "" or len(cascades)>=count:
            break

        values = line.strip('\n').split('\t')
        mid = values[0]
        try:
            url = values[1]
            user = users[url]
        except:
            continue
        root = cs.User(user.name, gender=user.gender, loc=user.loc, weibo_num=user.weibo_num, following_num=user.following_num, follower_num=user.follower_num, verification=user.verification, created_at=WTime(values[7]), text=values[8])
        forward_num = int(values[5])
        cascade = cs.Cascade(root, mid, comment=int(values[4]), forward=forward_num, like=int(values[6]), start_time=WTime(values[7]), type = type)
        while(True):
            forward_line = file.readline()
            
            line_num += 1
            forward_values = forward_line.strip('\n').split('\t')

            if forward_values[0] != mid:
                read = 1
                break
            if len(cascade.nodes) >= threshold:
                continue
            forward_name = forward_values[2]
            forward_created_at = WTime(forward_values[3])
            forward_text = forward_values[4]
            parents = trace_parents(forward_text)
            node = cs.Node(forward_name, created_at = forward_created_at, text=forward_text)

            if len(parents) == 0:
                cascade.add_child(node, cascade.root)
            else:
                for parent_node in cascade.nodes:
                    if parent_node.name == parents[0]:
                        cascade.add_child(node, parent_node)
                        break
        if len(cascade.nodes)>=10:
            cascade_num += 1
            cascades.append(cascade)
                
    file.close()
    return cascades

def trace_parents(forward_text):
    parents = []
    at_pattern = re.compile(u"\/\/<a href=\".*?@.*?<\/a>")
    for text in at_pattern.findall(forward_text):
        name_pattern = re.compile(u'@.*?<\/a>')
        name = name_pattern.search(text).group()[1:-4]
        parents.append(name)
    return parents

def load_users(filename, with_text = False):
    users = {}
    try:
        file = open(filename)
        for line in file:
            if line.startswith(codecs.BOM_UTF8):
                line = line[3:]
            values = line.strip('\n').split('\t')
            url = values[0]
            uid = int(values[1])
            name = values[2]
            user = cs.User(name, gender=values[3], loc=values[4], weibo_num=int(values[5]), following_num=int(values[6]), follower_num=int(values[7]), verification=int(values[8]))
            users[url] = user
    finally:
        file.close()
    return users

def get_user_from_name(name):
    for uid in users:
        if users[uid].name == name:
            return user

def utf8_to_ansi(input_file, output_file):
    input = open(input_file)
    output = open(output_file, 'w')
    count = 0
    for line in input:
        if line.startswith(codecs.BOM_UTF8):
            line = line[3:]
        try:
            line = line.decode('utf-8').encode('GBK')
        except:
            continue

        count += 1
    print count

