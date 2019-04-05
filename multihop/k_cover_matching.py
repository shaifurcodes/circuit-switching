#!/usr/bin/env python
'''
Module: k_cover_matching
List of entities:
Use:

'''
__author__ = "Md Shaifur Rahman"
__email__ = "mdsrahman@cs.stonybrook.edu"
__date__ = "April 04, 2019"
__license__ = "GPL"
__copyright__ = "Copyright 2019, WINGS Lab, Stony Brook University"

import numpy as np
from scipy.sparse import csgraph

#---start here--->
class MultiHopMatching(object):
    '''
    tasks:
    1. load data from file (packets+path)
    2. for each iteration, generate the traffic matrix and run single hop
    '''
    def __init__(self):
        '''

        '''
        self.topology = None
        self.n =  0  #no of nodes
        self.demand= None  #demand matrix, have to find shortest paths
        self.paths = None #list of (shortest) paths generated from topology for each pair of (source, dest)

    def parse_path(self, i, j, pred_matrix):
        '''

        :param i: source index
        :param j: dest index
        :param pred_matrix: predecessor matrix, ref: scipy.sparse.csmatrix.shortest_path
        :return: find the path i->j, else None if no path exists
        '''
        path = [j]
        while True:
            v = pred_matrix[i, j]
            if v < 0:
                return  None
            path.insert(0, v)
            if v==i:
                break
            j = v
        return path

    def generatePaths(self):
        '''

        :return:
        '''
        path_weight, path_pred = csgraph.shortest_path( self.topology, directed=True, unweighted=True, return_predecessors= True )
        self.paths = []
        for i in np.arange(self.n):
            for j in np.arange(self.n):
                if self.demand[i, j] >0:
                    path_i_j = self.parse_path(i, j, path_pred)
                    if path_i_j is None: continue #no path exists
                    #else
                    self.paths.append([i, j, path_i_j])
        print self.paths
        return

    def read_input(self, base_filename):
        '''

        :param base_filename: string, indicates two files, base_filename+"-input.txt" and base_filename+"-topology.txt"
        :return:
        '''
        #----first read the topology----#
        with open(base_filename+"-topology.txt") as f:
            for line in f:
                if line[0] == '#': continue
                words = line.split()
                if len(words) <=0: continue
                #--else process---#
                if words[0] == "complete":
                    # process complete graph
                    self.n = int( words[1] )
                    self.topology = np.ones((self.n, self.n), dtype=np.int )
                    break
                elif words[0] == "partial":
                    # process incomplete graph
                    self.n = int(words[1])
                    self.topology = np.zeros((self.n, self.n), dtype=np.int )
                else: #must be topology entry
                    i = int( words[0] ) -1
                    j_list = words[1].split(',')
                    for j in j_list:
                        j = int(j) - 1
                        self.topology[ i, int( j ) ] = 1
        print "topology:"
        print self.topology
        #----now read the traffic load---#
        self.demand = np.zeros((self.n, self.n), dtype=np.int)
        with open(base_filename+"-input.txt") as f:
            for line in f:
                if line[0] == '#': continue
                words = line.split(',')
                if len(words) <= 0: continue
                #---else process---#
                i, j, v = int(words[0])-1, int(words[1])-1, int(words[2])
                self.demand[i, j] = v
        #-----#
        print "demand:"
        print self.demand
        return
    #----end of class-----#

if __name__ == '__main__':
    '''
    module test
    '''
    base_filename = 'multipath'
    m = MultiHopMatching()
    m.read_input(base_filename=base_filename)
    m.generatePaths()

    #---end of main---#