#!/usr/bin/env python
'''
Module: matching
List of entities:
Use: Contains classes, methods for matching algorithms

'''
__author__ = "Md Shaifur Rahman"
__email__ = "mdsrahman@cs.stonybrook.edu"
__date__ = "April 03, 2019"
__license__ = "GPL"
__copyright__ = "Copyright 2019, WINGS Lab, Stony Brook University"

import numpy as np
from scipy.optimize import linear_sum_assignment

class Sigmetric1HopMatching(object):
    '''

    '''
    def __init__(self, connection, traffic, switching_delay, window_size, debug = False):
        '''
        :param connection: nxn bool matrix, for (i, j) directional connectivity
        :param traffic: nxn float matrix, for (i, j) directional traffic demand
        :param switching_delay: float,
        :param window_size: float, should be >= switching_delay
        :param debug: bool, whether to turn on additional book-keeping for debug
        '''
        self.connection = connection
        self.traffic = traffic
        self.switching_delay = switching_delay
        self.window_size = window_size
        #------------------#
        self.DEBUG = debug
        self.n, _ = connection.shape
        self.rem_traffic = np.where(connection, traffic, 0.)
        self.current_duration = None
        self.schedule = [] # array of tuple ( duration (float), array of n_i, array of n_j ) s.t. n_i -> n_j in the matching
        if self.DEBUG:
            self.debug_rem_traffic_list = []
        self.FRACTION_OF_TRAFFIC = 1


    def get_score_efficient_util(self, duration):
        '''
        pre-condition: self.rem_traffic must be set
        :param duration: chosen duration
        :return: matching score (float), n_i list, n_j list such that n_i -> n_j matched
        '''

        edge_weights = np.clip( self.rem_traffic, 0.0, duration )
        row_indx, col_indx = linear_sum_assignment( - edge_weights )
        total_packets = np.sum( edge_weights[row_indx, col_indx] )
        matching_score = total_packets / (1.0*duration + self.switching_delay)
        return matching_score, row_indx, col_indx


    def get_duration_efficient_util(self, max_duration):
        '''

        :return:
        '''
        best_score = -1.0
        best_duration = None
        best_row_indx = None
        best_col_indx = None
        demand_set = self.rem_traffic.flatten()
        for demand in demand_set: #TODO: make a binary search here
            if demand <= 0. or demand > max_duration: #ignore the duration <=0 or > remaining_duration
                continue
            cur_score, cur_row, cur_col = self.get_score_efficient_util(demand)
            #print "debug: current_score, duration ", cur_score, demand
            if cur_score > best_score:
                best_duration, best_score, best_row_indx, best_col_indx =  demand, cur_score, cur_row, cur_col
        if self.DEBUG:
            print "debug chosen (best score, duration): ", best_score, best_duration
        return best_duration, best_row_indx, best_col_indx


    def iterative_matching(self):
        '''
        :return:
        '''
        sum_of_duration = 0.
        while sum_of_duration < self.window_size:
            remaining_duration = self.window_size - sum_of_duration
            current_duration, row_indx, col_indx = self.get_duration_efficient_util( remaining_duration )
            if ( current_duration <= 0. ) or  \
                    ( current_duration+sum_of_duration > self.window_size ):
                break    #cannot accommodate current duration, either zero or exceeds window
            #else:
            #save the matching, update the demand matrix
            self.schedule.append((current_duration, row_indx, col_indx))
            self.rem_traffic[ row_indx, col_indx ] = np.clip( (self.rem_traffic[ row_indx, col_indx ] - current_duration),a_min=0., a_max = None)
            if self.DEBUG:
                self.debug_rem_traffic_list.append(np.copy(self.rem_traffic))
            sum_of_duration += current_duration
        return

    def get_metric(self, metric_name):
        '''

        :param metric_name: must of one of self.FRACTION_OF_TRAFFIC,..
        :return:
        '''
        if metric_name == self.FRACTION_OF_TRAFFIC:
            total_demand = np.sum(self.traffic)
            demand_met = np.sum( self.rem_traffic )
            fraction_of_demand_met = (total_demand - demand_met) / (1.*total_demand)
            return  fraction_of_demand_met
        #else:
        return 0.
    #----end of class Matching-----#

    def print_schedule(self):
        '''
        :return:
        '''
        print "given demand:"
        print self.traffic
        total_duration  = 0.
        for i, v in enumerate(self.schedule):
            alpha, n_i_list, n_j_list = v
            print "============================="
            print i+1, " duration:", alpha
            total_duration += alpha
            for n_i, n_j in zip(n_i_list, n_j_list):
                print "\t",n_i,"->",n_j
            if self.DEBUG:
                print "sum_alpha/W:",total_duration,"/",self.window_size,"remaining demand"
                print self.debug_rem_traffic_list[i]
        return


if __name__ == '__main__':
    '''
    module test
    '''
    switching_delay = 2.
    window_size = 12.
    swtopo = np.full((3, 3), True, dtype=np.bool)

    #swtopo[ 1, 0] = False
    tmat = np.array( [ [ 0., 5., 6.],
                       [ 18., 0., 0.],
                       [ 1., 4., 0.]
                     ], dtype = np.float )

    debug = True #<--debug switch, turn on additional data-saving
    m = Sigmetric1HopMatching(connection=swtopo, traffic=tmat, switching_delay=switching_delay, window_size = window_size, debug = debug)
    m.iterative_matching()
    m.print_schedule()
    print "utilization: ", np.round( 100.* m.get_metric(m.FRACTION_OF_TRAFFIC), 2 ), "%"

    #-----end of module-----#
