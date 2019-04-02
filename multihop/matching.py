#!/usr/bin/env python
'''
Module: matching
List of entities:
Use: Contains classes, methods for matching algorithms

'''
__author__ = "Md Shaifur Rahman"
__email__ = "mdsrahman@cs.stonybrook.edu"
__date__ = "April 02, 2019"
__license__ = "GPL"
__copyright__ = "Copyright 2019, WINGS Lab, Stony Brook University"

import numpy as np
from scipy.optimize import linear_sum_assignment

class Matching(object):
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

    def bipartite_matching(self):
        '''
            pre-condition: must set self.current_duration and self.rem_traffic first
            post-condition: i) matching saveind in self.schedule
                            ii) self.rem_traffic is possibly changed
        :return: True if non-zero demand met after matching, else false
        '''

        #do the munkres assignment
        row_indx, col_indx = linear_sum_assignment( -self.rem_traffic )

        if np.sum( self.rem_traffic[row_indx, col_indx] ) <= 0.:
            return  False # no demand to meet
        #else some demand to meet

        #save the matching
        self.schedule.append((self.current_duration, row_indx, col_indx))

        #update the remaining traffic
        self.rem_traffic[ row_indx, col_indx ] = (self.rem_traffic[ row_indx, col_indx ] - self.current_duration).clip(min = 0.)
        if self.DEBUG:
            self.debug_rem_traffic_list.append(np.copy( self.rem_traffic) )
        return True

    def find_current_duration(self):
        '''
        simplest strategy is to take average of the remaining demands
        TODO: implement effective utilization metrics here for sigmetrics paper
        :return:
        '''
        non_zero_demands = self.rem_traffic[np.where(self.rem_traffic > 0.)]
        if non_zero_demands.size == 0.:
            return  0.
        d = int( np.median( non_zero_demands ) )
        return d

    def iterative_matching(self):
        sum_of_duration = 0.
        while sum_of_duration < self.window_size:
            self.current_duration = self.find_current_duration()
            if self.current_duration <= 0. or  self.current_duration+sum_of_duration > self.window_size:
                break    #cannot accommodate current duration, either zero or exceeds window
            if not self.bipartite_matching():  # no demand to meet, the remaining remaining traffic unchanged
                break
            sum_of_duration += self.current_duration
        return
    #----end of class Matching-----#

    def print_schedule(self):
        '''
        :return:
        '''
        print "given demand:"
        print self.traffic
        for i, v in enumerate(self.schedule):
            alpha, n_i_list, n_j_list = v
            print i+1, " duration:", alpha
            for n_i, n_j in zip(n_i_list, n_j_list):
                print "\t",n_i,"->",n_j
            if self.DEBUG:
                print "remaining demand"
                print self.debug_rem_traffic_list[i]
        return


if __name__ == '__main__':
    '''
    module test
    '''
    switching_delay = 5.
    window_size = 9.
    swtopo = np.full((3, 3), True, dtype=np.bool)

    #swtopo[ 1, 0] = False
    tmat = np.array( [ [ 0., 2., 2.],
                       [ 8., 0., 0.],
                       [ 1., 2., 0.]
                     ], dtype = np.float )

    debug = True #<--debug switch, turn on additional data-saving
    m = Matching(connection=swtopo, traffic=tmat, switching_delay=switching_delay, window_size = window_size, debug = debug)
    m.iterative_matching()
    m.print_schedule()

    #-----end of module-----#
