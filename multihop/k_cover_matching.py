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
from scipy.optimize import linear_sum_assignment

from utility_matching import get_one_hop_matching_score
#---start here--->
class MultiHopMatching(object ):
    '''
    tasks:
    1. load data from file (packets+path)
    2. for each iteration, generate the traffic matrix and run single hop
    '''
    def __init__(self, window_size, switching_delay):
        '''

        '''
        self.switching_delay = switching_delay
        self.window_size = window_size
        #-----------#
        self.total_duration = 0
        self.topology = None
        self.n =  0  #no of nodes
        self.demand= None  #demand matrix, have to find shortest paths
        self.current_demand = None #curernt demand matrix, entries become zero when packet reaches destination
        self.cur_edge_weights = None
        self.node_demands = None
        self.demand_met = None

        #self.path_lengths = None # keeps the length of the paths

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
        self.node_demands = []
        for i in np.arange(self.n):
            self.node_demands.append([])

        path_weight, path_pred = csgraph.shortest_path( self.topology, directed=True, unweighted=True, return_predecessors= True )

        for i in np.arange(self.n):
            for j in np.arange(self.n):
                if self.demand[i, j] >0:
                    path_i_j = self.parse_path(i, j, path_pred)
                    if path_i_j is None: continue #no path exists
                    #else
                    self.node_demands[i].append([ self.demand[i, j], path_i_j ])
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

    def read_input_multipath(self, base_filename):
        '''

        :param base_filename:
        :return:
        '''
        #TODO: add multipath
        return

    def add_path_to_node_demand(self, data, node_index, demand, path):
        data[node_index].append([demand, path])
        return data

    def init_node_demand(self):
        node_demand  = []
        for i in np.arange(self.n):
            node_demand.append([])
        return  node_demand

    def clear_demand(self, duration, row_indx, col_indx):
        matching_capacity = np.zeros( (self.n, self.n), dtype=np.float)
        new_node_demands = self.init_node_demand()
        for cur_node, cur_node_demands in enumerate(self.node_demands):  # iterate over demands of all nodes
            for indx, demand_path in enumerate( cur_node_demands) :
                demand, path = demand_path[0], demand_path[1]
                i, j = path[0], path[1]
                isMatchingFound = False
                for (n1, n2) in zip(row_indx, col_indx):
                    if n1 == i and n2 == j and matching_capacity[i, j] < duration:
                        isMatchingFound = True
                        remaining_capacity = duration - matching_capacity[i, j]
                        if remaining_capacity >= demand:
                            # if not destination, add demand_path to the new edge, update matching capacity
                            matching_capacity[i, j] += demand
                            if len(path) <=2: #ensures that this is the last hop
                                self.demand_met[i, j] += demand #TODO: for book-keeping save the (demand_met, path) for this iteration
                            else:
                                new_node, new_demand, new_path = j, demand, path[1:]
                                new_node_demands = self.add_path_to_node_demand(new_node_demands, new_node, new_demand, new_path)
                            break
                        else:
                            #if not deistination, a
                            prev_node, prev_demand, prev_path = i, (demand - remaining_capacity), path
                            new_node_demands = self.add_path_to_node_demand(new_node_demands, prev_node, prev_demand, prev_path)
                            if len(path) <=2:
                                self.demand_met[i, j] += remaining_capacity #TODO: for book-keeping save the (demand_met, path) for this iteration
                            else:
                                new_node, new_demand, new_path = j, remaining_capacity, path[1:]
                                new_node_demands = self.add_path_to_node_demand(new_node_demands, new_node, new_demand, new_path)
                            break
                if not isMatchingFound: #no mathcing exist in the current iteration for this path, so add it too
                    prev_node, prev_demand, prev_path = i, demand , path
                    new_node_demands = self.add_path_to_node_demand(new_node_demands, prev_node, prev_demand, prev_path)

        self.node_demands = new_node_demands
        return

    def set_edge_weights(self):
        '''
            for each element in the path_list calulcate weight
        :return:
        '''
        self.cur_edge_weights = np.zeros( (self.n, self.n), dtype=np.float  )
        for cur_node, cur_node_demands in enumerate( self.node_demands) : #iterate over demands of all nodes
            for val in cur_node_demands:
                node_demand, node_path = val[0], val[1]
                if node_demand <= 0 or len(node_path)<=0 :
                    continue
                if node_path[0] != cur_node:
                    print "Error: cur_node != path_i !!" #check: first node in the path must be current( initiating node )
                i, j = node_path[0], node_path[-1]
                self.cur_edge_weights[i, j] += node_demand / (1.*(len(node_path) - 1))

        # extra check, unconnected edges get zero weights
        self.cur_edge_weights = np.where(self.topology == 0, 0, self.cur_edge_weights)
        return

    def decorated_print_node_demands(self, heading_txt = None ):
        '''

        :param heading_txt:
        :return:
        '''
        if heading_txt is not None:
            #print "===================================="
            print heading_txt
            print "===================================="
        for i in np.arange(self.n):
            if len(self.node_demands[i]) > 0:
                print "Node # ", i
                for val in self.node_demands[i]:
                    d, p = val
                    first_node, last_node = p[0], p[-1]
                    print "\t", first_node," --> ",last_node
                    print "\t\tdemand: ",d,"\t",
                    print  "path: ",
                    for cur_node in p[0:-1]:
                        print cur_node, "-->",
                    print p[-1]
        print "===================================="
        return

    def show_demand_met(self):
        if self.demand_met is None:
            print "demand_met is empty!!"
            return
        total_demand_met = np.sum(self.demand_met)
        total_input_demand = np.sum(self.demand)
        print "Demand met (%): ", 100.0*(total_demand_met/total_input_demand)
        return

    def iterative_matching(self):
        '''
        1. for each iteration
            i) calculate the Traffic_rem
            ii) calculate (\alpha, M)
            iii) update
        :return:
        '''
        self.demand_met = np.zeros((self.n, self.n), dtype=np.float )
        self.current_duration = 0
        self.current_demand = np.copy( self.demand )
        #print "init node demands:"
        #print self.node_demands
        iteration_count = 1
        while self.current_duration < self.window_size :
            self.decorated_print_node_demands("Iteration # " + str(iteration_count))
            self.set_edge_weights()
            max_duration = self.window_size - self.current_duration
            duration, row_indx, col_indx = get_one_hop_matching_score( switching_delay = self.switching_delay,
                                                                            bipartite_edge_weights = self.cur_edge_weights,
                                                                            max_duration = max_duration,
                                                                            min_duration = 0.)
            if duration is None:
                break
            #else, clear up demands, based on this matching
            duration = int(np.floor(duration)) #cast to integer
            self.current_duration += duration
            self.clear_demand( duration=duration, row_indx=row_indx, col_indx=col_indx)

            iteration_count += 1
        return

    #----end of class-----#

if __name__ == '__main__':
    '''
    module test
    '''
    window_size, switching_delay = 100, 4
    base_filename = 'multipath'
    m = MultiHopMatching(window_size=window_size, switching_delay=switching_delay)
    #---------------------#
    #TODO: merge the following two for simplicity to generate random multi-path
    m.read_input(base_filename=base_filename)
    m.generatePaths()
    #--------------#
    m.iterative_matching()
    m.show_demand_met()

    #---end of main---#