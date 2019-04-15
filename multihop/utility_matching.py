#!/usr/bin/env python
'''
Module: utility_matching
List of entities:
Use:

'''
__author__ = "Md Shaifur Rahman"
__email__ = "mdsrahman@cs.stonybrook.edu"
__date__ = "April 05, 2019"
__license__ = "GPL"
__copyright__ = "Copyright 2019, WINGS Lab, Stony Brook University"

import numpy as np
from scipy.optimize import linear_sum_assignment

#---start here--->


def get_one_hop_matching_score( switching_delay, bipartite_edge_weights, max_duration, min_duration = 0. ):
    '''

    :param switching_delay: float, should be less than the window size
    :param bipartite_edge_weights: edge weights for the bipartite matching, unconnected edges should be of 0. value
    :param max_duration: only duration values <= max_duration are considered
    :param min_duration: only duration values > min_duration are considered
            this method considers only values from the bipartite_edge_weights limited by the [max_duration, min_duration)
    :return: duration, matching row indices, matching column indices,
            check duration value 'None', indicating no matching operation was possible
    '''
    best_score = -1.0
    best_duration = None
    best_row_indx = None
    best_col_indx = None
    init_duration_set = bipartite_edge_weights.flatten()
    duration_set = []
    for i in init_duration_set:
        if i >0.0 and ( i not in duration_set ):
            duration_set.append(i)
    #print "debug: @get_one_hop_(..): ", duration_set

    for duration in duration_set:  # TODO: make a binary search here
        if duration <= min_duration or duration > max_duration:  # ignore the duration <=0 or > remaining_duration
            continue
        clipped_weights =  np.clip( bipartite_edge_weights, a_max=duration, a_min=0 )
        row_indx, col_indx = linear_sum_assignment(- clipped_weights)
        matching_score = np.sum(clipped_weights[row_indx, col_indx]) / (1.0 * duration + switching_delay)
        # print "cur weights: "
        # print clipped_weights
        # print "debug: ", duration, matching_score, row_indx, col_indx
        # print "======================="
        if matching_score > best_score:
            best_duration, best_score, best_row_indx, best_col_indx = duration, matching_score, row_indx, col_indx
    return best_duration, best_row_indx, best_col_indx

if __name__ == '__main__':
    '''
    module test
    '''
    switching_delay = 1.
    swtopo = np.full((3, 3), True, dtype=np.bool)
    tmat = np.array( [ [ 0., 2., 6.],
                       [ 12., 0., 16.],
                       [ 1., 7., 0.]
                     ], dtype = np.float )
    d, r, c =  get_one_hop_matching_score(max_duration=20,
                                          min_duration=0 ,
                                          switching_delay = switching_delay,
                                          bipartite_edge_weights=tmat)
    print d, r, c
