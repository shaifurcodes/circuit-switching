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

#TODO: given a residual traffic matrix T_rem, switching topology S_w, compute assignment for a given duration

class Matching(object):
    '''

    '''
    def __init__(self, connection, traffic):
        '''

        :param connection: nxn bool matrix, for (i, j) directional connectivity
        :param traffic: nxn float matrix, for (i, j) directional traffic demand
        '''
        self.connection = connection
        self.traffic = traffic
        self.rem_traffic = np.copy(traffic)
        self.matching = None
        self.n, _ = connection.shape

        self.UNASSIGNED = -1

    def single_hop_matching(self):
        '''
        :return:
        '''
        #Pre-screen connectivity and traffic demand
        cur_matching_weight = -np.where(self.connection, self.rem_traffic, 0.)
        #do the munkres assignment
        row_indx, col_indx = linear_sum_assignment( cur_matching_weight )
        #update the remaining traffic
        self.rem_traffic[ row_indx, col_indx ] = self.rem_traffic[ row_indx, col_indx ] \
                                                 -  self.rem_traffic[ row_indx, col_indx ]

        #TODO: save the matching
        self.matching = np.full((self.n), self.UNASSIGNED, dtype=np.int )
        self.matching[ row_indx ] = col_indx
        return

    def print_current_matching(self):
        '''

        :return:
        '''
        for i, v in np.ndenumerate(self.matching):
            print i,"-->",v

        print "remaining traffic:"
        print self.rem_traffic
        return

if __name__ == '__main__':
    '''
    module test
    '''
    swtopo = np.full((3, 3), True, dtype=np.bool)

    swtopo[ 1, 0] = False
    tmat = np.array( [ [ 0., 3., 4.],
                       [ 6., 0., 2.],
                       [ 5., 2., 0.]
                     ], dtype = np.float )

    m = Matching(connection=swtopo, traffic=tmat)
    m.single_hop_matching()
    m.print_current_matching()

    #-----end of module-----#
