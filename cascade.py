import networkx as nx
import re
from datetime import datetime
import os.path, os, random

cascades = {}
cascade_count = 0

class Node(object):
    def __init__(self, name, created_at = '', text = ''):
        self.name  = name
        self.created_at = created_at
        self.text = text
        self.depth = 1
        self.children = []
        
    def add_child(self, obj):
        obj.depth = self.depth + 1
        obj.parent = self
        self.children.append(obj)

class User(Node):
    def __init__(self, name, gender='', loc='', weibo_num=0, following_num=0, follower_num=0, verification=0, created_at = '', text = ''):
        Node.__init__(self, name, created_at=created_at, text=text)
        self.gender = gender
        self.loc = loc
        self.weibo_num = weibo_num
        self.following_num = following_num
        self.follower_num = follower_num
        self.verification = verification
        self.parent = None

class Cascade():
    def __init__(self, root, mid, type='normal', start_time=None, comment=0, forward=0, like=0):
        self.root = root
        self.mid = mid
        self.size = 1
        self.depth = 1

        self.type = type
        self.start_time = start_time
        self.comment = comment
        self.forward = forward
        self.like = like
        
        self.nodes=[root]
       
    def degree(self, node):
        d = len(node.children)
        #if node == self.root:
        #    d += 1
        return d

    def add_node(self, node):
        self.nodes.append(node)
        self.size += 1
        
    def add_child(self, node, parent):
        self.add_node(node)

        parent.add_child(node)
        if self.depth <= node.depth:
            self.depth = node.depth

    def find_parent(self, node):
        for parent in self.nodes:
            for child in parent.children:
                if child.name == node.name:
                    return parent

    def find(self, name):
        for node in self.nodes:
            if node.name == name:
                return node


    def save(self, path):
        '''save according to time'''
        filename = self.mid + '_' + str(self.size) + '_' + str(self.depth) + '.txt'
        file = open(os.path.join(path, filename), 'w')

        sorted_nodes = sorted(self.nodes, key = lambda x: x.created_at)
        file.write(','+self.root.uid+','+self.root.created_at.strftime("%Y-%m-%d %H:%M:%S")+','+self.root.text+'\n')
        for node in sorted_nodes:
            if node.uid != self.root.uid:
                file.write(node.parent.uid+','+node.uid+','+node.created_at.strftime("%Y-%m-%d %H:%M:%S")+','+node.text+'\n')

        file.close()
    
    def print_cascade(self):
        for node in self.nodes:
            print '\t', node.name, node.depth, node.created_at.mt-node.created_at.mt


