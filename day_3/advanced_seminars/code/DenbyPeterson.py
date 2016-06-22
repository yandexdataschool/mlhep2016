__author__ = 'mikhail91'

import numpy
import pandas
import matplotlib.pyplot as plt

class DenbyPeterson(object):

    def __init__(self,
                 n_iter,
                 cos_degree=1,
                 alpha=0.01,
                 delta=0.001,
                 temperature=1000,
                 temperature_decay_rate=0.99,
                 max_cos = -0.9,
                 state_threshold = 0.5,
                 min_hits=3,
                 save_stages=True):
        """
        This class is simple realization of the Denby-Peterson method of tracks recognition.
        :param n_iter: int, number of iteration.
        :param cos_degree: int, degree of the cos value for the angle between the two neurons.
        :param alpha: float, multiplier of the penalty function against bifurcations.
        :param delta: float, multiplier of the penalty function to balance number of active neurons against number of hits.
        :param temperature: float, parameter in the neuron's state updating rule.
        :param temperature_decay_rate: float, decay rate of the temperature during the network optimization.
        :param max_cos: float, max cos value for the angle between the two neurons.
        The neurons with larger values will be removed after the network optimization.
        :param state_threshold: float, is sate of a neuron is greater than this value, the neuron will marked as active.
        :param min_hits: int, min number of hits in a track candidate.
        :param save_stages: boolean, if True, the neurons states will be saved after each iteration.
        :return:
        """

        self.n_iter = n_iter
        self.cos_degree = cos_degree
        self.alpha = alpha
        self.delta = delta
        self.temperature = temperature
        self.temperature_decay_rate = temperature_decay_rate
        self.max_cos = max_cos
        self.state_threshold = state_threshold
        self.min_hits = min_hits
        self.save_stages = save_stages

        self.states_stages_ = []
        self.energy_stages_ = []

        self.states_ = None
        self.labels_ = None
        self.states_after_cut_ = None

    def calc_energy(self, num_hits, weights, states):
        """
        This function calculates energy of the network.
        :param num_hits: int, number of hits.
        :param weights: numpy.ndarray shape = [num_hits, num_hits, num_hits], weights for each pair of neurons in the cost function.
        :param states: numpy.ndarray shape = [num_hits, num_hits], states of the neurons.
        :return: float, energy value.
        """

        energy = 0
        for i in range(num_hits):

            for j in range(num_hits):

                for k in range(num_hits):

                    energy += - 0.5 * weights[i, j, k] * states[i, j] * states[j, k]

                    if j != k:

                        energy += 0.5 * self.alpha * states[i, j] * states[i, k]

                    if i != k:

                        energy += 0.5 * self.alpha * states[i, j] * states[k, j]

        energy += 0.5 * self.delta * (states.sum() - num_hits)**2

        return energy

    def calc_delta_energy(self, num_hits, weights, states):
        """
        This function calculates derivative of the energy function.
        :param num_hits: int, number of hits.
        :param weights: numpy.ndarray shape = [num_hits, num_hits, num_hits], weights for each pair of neurons in the cost function.
        :param states: numpy.ndarray shape = [num_hits, num_hits], states of the neurons.
        :return: float, energy value.
        """

        delta_energy = numpy.zeros((num_hits, num_hits))

        for i in range(num_hits):

            for j in range(num_hits):

                for k in range(num_hits):

                    delta_energy[i, j] += - 0.5 * weights[i, j, k] * states[j, k]

                    if j != k:

                        delta_energy[i, j] += 0.5 * self.alpha * states[i, k]

                    if i != k:

                        delta_energy[i, j] += 0.5 * self.alpha * states[k, j]

        delta_energy[i, j] += self.delta * (states.sum() - num_hits)

        return delta_energy

    def cut_neurons(self, x, y, states, max_cos, state_threshold):
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

    def find_neighbor(self, first_point, states, state_threshold):
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

    def find_tracks(self, states, state_threshold):
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

                        new_start = self.find_neighbor(start, states_copy, state_threshold)
                        states_copy[start, new_start] = states_copy[new_start, start] = 0
                        labels[start] = track_id
                        start = new_start

                    while end != -1:

                        new_end = self.find_neighbor(end, states_copy, state_threshold)
                        states_copy[end, new_end] = states_copy[new_end, end] = 0
                        labels[end] = track_id
                        end = new_end

                    track_id += 1

        return labels

    def cut_labels(self, labels, min_hits=3):
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

    def fit(self, x, y):
        """
        This function runs the Denby-Peterson method.
        :param x: numpy.array shape = [n_hits], array of x-coordinates of the hits.
        :param y: numpy.array shape = [n_hits], array of y-coordinates of the hits.
        :return:
        """

        num_hits = len(x)

        # Distances
        dist = numpy.zeros((num_hits, num_hits))

        for i in range(num_hits):

            for j in range(num_hits):

                r = numpy.sqrt( (x[i] - x[j])**2 + (y[i] - y[j])**2 )
                dist[i, j] = r

        # Weights
        weights = numpy.zeros((num_hits, num_hits, num_hits))
        for i in range(num_hits):

            for j in range(num_hits):

                for k in range(num_hits):

                    if i==j or i==k or j==k:
                        continue

                    scalar_prod = (x[i] - x[j])*(x[k] - x[j]) + (y[i] - y[j])*(y[k] - y[j])
                    cos = scalar_prod / (dist[i,j] * dist[j, k])
                    weights[i,j,k] = - cos**self.cos_degree / (dist[i,j] + dist[j, k])

        # States
        states = numpy.random.rand(num_hits, num_hits)
        states = 0.5 * (states + states.T)

        # Energy
        energy = self.calc_energy(num_hits, weights, states)

        # Delta energy
        if self.save_stages:

            self.states_stages_.append(states)
            self.energy_stages_.append(energy)


        for one_iter in range(self.n_iter):

            delta_energy = self.calc_delta_energy(num_hits, weights, states)
            states = 0.5 * (1 + numpy.tanh( - delta_energy / self.temperature))
            states = 0.5 * (states + states.T)
            energy = self.calc_energy(num_hits, weights, states)

            self.temperature *= self.temperature_decay_rate

            if self.save_stages:

                self.states_stages_.append(states)
                self.energy_stages_.append(energy)


        states_after_cut = self.cut_neurons(x, y, states, self.max_cos, self.state_threshold)
        labels = self.find_tracks(states_after_cut, self.state_threshold)
        labels = self.cut_labels(labels, self.min_hits)

        self.labels_ = labels
        self.states_after_cut_ = states_after_cut





def plot_neural_net(x, y, states, state_threshold):
    """
    This function plots the network.
    :param x: numpy.array shape = [n_hits], array of x-coordinates of the hits.
    :param y: numpy.array shape = [n_hits], array of y-coordinates of the hits.
    :param states: numpy.ndarray shape = [num_hits, num_hits], states of the neurons.
    :param state_threshold: float, is sate of a neuron is greater than this value, the neuron will marked as active.
    :return:
    """

    N = len(x)

    plt.figure(figsize=(10,7))
    plt.scatter(x, y)

    for i in range(N):
        for j in range(N):

            if states[i, j] > state_threshold:
                plt.plot([x[i], x[j]], [y[i], y[j]], color='0.5', alpha=0.5)

    plt.xlabel('X', size=15)
    plt.ylabel('Y', size=15)
    plt.xticks(size=15)
    plt.yticks(size=15)



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
