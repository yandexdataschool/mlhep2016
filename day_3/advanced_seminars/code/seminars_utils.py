__author__ = 'mikhail91'

import numpy
import pandas

def rotors2states(x, y, rotor, min_cos):

    n_hits = len(x)
    states = numpy.zeros((n_hits, n_hits))

    for i in range(n_hits):

        min_dist = -1
        min_j = -1

        for j in range(n_hits):

            if i <= j or x[j] == x[i]:
                continue

            dist_v = [ x[j] - x[i], y[j] - y[i] ]
            dist_mod = numpy.sqrt((dist_v[0])**2 + (dist_v[1])**2)

            hit_rotor_v = rotor[i]
            hit_rotor_mod = numpy.sqrt((hit_rotor_v[0])**2 + (hit_rotor_v[1])**2)

            cos = numpy.inner(dist_v, hit_rotor_v) / (dist_mod * hit_rotor_mod)
            cos = numpy.abs(cos)

            if min_dist == -1:

                min_dist = dist_mod
                min_j = j

            if cos >= min_cos:

                if dist_mod < min_dist:

                    min_dist = dist_mod
                    min_j = j

        if states[i, :].sum() == 0 and states[:, min_j].sum() == 0:
            states[i, min_j] = 1

    return states + states.T

def cut_neurons(x, y, states, max_cos, state_threshold):
    """
    This function cuts neurons leaving just two neurons (input and output) with min cos values of the angle between them for one neuron.
    :param x: numpy.array shape = [n_hits], array of x-coordinates of the hits.
    :param y: numpy.array shape = [n_hits], array of y-coordinates of the hits.
    :param states: numpy.ndarray shape = [num_hits, num_hits], states of the neurons.
    :param max_cos: float, max cos value for the angle between the two neurons.
    The neurons with larger values will be removed after the network optimization.
    :param state_threshold: float, is sate of a neuron is greater than this value, the neuron will marked as active.
    :return: updated states, numpy.ndarray shape = [num_hits, num_hits], states of the neurons.
    """

    states_copy = states.copy()
    num_hits = len(states_copy)

    # Distances
    dist = numpy.zeros((num_hits, num_hits))

    for i in range(num_hits):

        for j in range(num_hits):

            r = numpy.sqrt( (x[i] - x[j])**2 + (y[i] - y[j])**2 )
            dist[i, j] = r


    for i in range(num_hits):

        for j in range(num_hits):

            min_k = -1
            min_cos = 2

            for k in range(num_hits):

                if i==j or i==k or j==k:
                    continue

                if states_copy[i,j] > state_threshold and  states_copy[j,k] > state_threshold:

                    scalar_prod = (x[i] - x[j])*(x[k] - x[j]) + (y[i] - y[j])*(y[k] - y[j])
                    cos = scalar_prod / (dist[i,j] * dist[j, k])

                    if cos < min_cos:

                        if min_k != -1:

                            states_copy[j, min_k] = states_copy[min_k, j] = 0
                            #states_copy[j, min_k] = 0

                        min_k = k
                        min_cos = cos

                    else:

                        states_copy[j, k] = states_copy[k, j] = 0
                        #states_copy[j, k] = 0

            if min_k != -1 and min_cos >= max_cos:

                states_copy[j, min_k] = states_copy[min_k, j] = 0
                #states_copy[j, min_k]  = 0

    return states_copy

def find_neighbor(first_point, states, state_threshold):
    """
    This function is searching for a hit connected with input one by an active neuron.
    :param first_point: int, index of a hit.
    :param states: numpy.ndarray shape = [num_hits, num_hits], states of the neurons.
    :param state_threshold: float, is sate of a neuron is greater than this value, the neuron will marked as active.
    :return: int, index of the paired hit.
    """

    num_hits = len(states)

    for second_point in range(num_hits):

        if states[first_point, second_point] > state_threshold and first_point != second_point:

            return second_point

    return -1

def find_tracks(states, state_threshold):
    """
    This function is searching for the tracks candidated.
    :param states: numpy.ndarray shape = [num_hits, num_hits], states of the neurons.
    :param state_threshold: float, is sate of a neuron is greater than this value, the neuron will marked as active.
    :return: labels, numpy.array shape = [n_hits], -1 meant, that a hit is noise.
    """

    states_copy = states.copy()
    num_hits = len(states_copy)

    labels = -1 * numpy.ones(num_hits)
    track_id = 0

    for i in range(num_hits):

        for j in range(num_hits):

            if states_copy[i, j] > state_threshold:

                start = i
                end = j

                labels[i] = track_id
                labels[j] = track_id

                states_copy[i, j] = states_copy[j, i] = 0

                while start != -1:

                    new_start = find_neighbor(start, states_copy, state_threshold)
                    states_copy[start, new_start] = states_copy[new_start, start] = 0
                    labels[start] = track_id
                    start = new_start

                while end != -1:

                    new_end = find_neighbor(end, states_copy, state_threshold)
                    states_copy[end, new_end] = states_copy[new_end, end] = 0
                    labels[end] = track_id
                    end = new_end

                track_id += 1

    return labels

def cut_labels(labels, min_hits=3):
    """
    This function cuts labels. If a track candidate has to small number of hits, this candidate is removed.
    :param labels: numpy.array shape = [n_hits], labels of the hits.
    :param min_hits: int, min number of hits in a track.
    :return: labels: numpy.array shape = [n_hits], labels of the hits.
    """

    new_labels = labels.copy()

    unique, counts = numpy.unique(labels, return_counts=True)
    for lab in unique[counts < min_hits]:

        new_labels[new_labels == lab] = -1

    return new_labels
