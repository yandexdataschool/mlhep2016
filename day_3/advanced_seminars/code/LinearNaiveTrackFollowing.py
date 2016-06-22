__author__ = 'mikhail91'
import numpy
import pandas
from copy import copy

from sklearn.linear_model import LinearRegression

class LinearNaiveTrackFollowing(object):

    def __init__(self, window, min_hits=4, n_last_fit=2):
        """
        This calss is simple realization of the Naive Track Following algorithm the straight track in 2D.
        :param window: float, if the difference between a hit and the extrapolation function is larger than thi value,
        the hit is rejected.
        :param min_hits: int, min number of hits in a track.
        :param n_last_fit: int, number of last point used for the extrapolation.
        :return:
        """

        self.window = window
        self.min_hits = min_hits
        self.n_last_fit = n_last_fit
        self.labels_ = None
        self.predictions_ = None

    def fit(self, x, y):
        """
        This finction runs the Naive Track Following algorithm.
        :param x: numpy.array shape = [n_hits], array of x-coordinates of the hits.
        :param y: numpy.array shape = [n_hits], array of y-coordinates of the hits.
        :return:
        """

        self.labels_ = -1 * numpy.ones(len(x))
        self.predictions_ = copy(y)

        layers_x = numpy.unique(x)
        layers = numpy.arange(len(layers_x))
        indeces = numpy.arange(len(x))

        track = []
        track_id = 0

        # Loop over all seeds
        for first_i in indeces[x == layers_x[0]]:
            for second_i in indeces[x == layers_x[1]]:

                if self.labels_[first_i] != -1 or self.labels_[second_i] != -1:
                    continue

                track = []
                track_predictions = []
                track += [first_i, second_i]
                track_predictions += [y[first_i], y[second_i]]

                # Loop over all layers
                for layer_id in layers[2:]:

                    if self.n_last_fit == 'all':
                        lr = LinearRegression()
                        lr.fit(x[track].reshape(-1,1), y[track])
                    else:
                        lr = LinearRegression()
                        lr.fit(x[track[-self.n_last_fit:]].reshape(-1,1), y[track[-self.n_last_fit:]])

                    curr_layer_x = layers_x[layer_id]
                    y_pred = lr.predict([[curr_layer_x]])[0]

                    ys_in_layer = y[x == layers_x[layer_id]]
                    indeces_in_layer = indeces[x == layers_x[layer_id]]

                    dys_in_layer = numpy.abs(ys_in_layer - y_pred)
                    dys_in_layer_min = dys_in_layer.min()

                    if dys_in_layer_min <= self.window:

                        hit_indexes = indeces_in_layer[dys_in_layer == dys_in_layer_min]
                        for hit_index in hit_indexes:
                            if self.labels_[hit_index] == -1:
                                track += list([hit_index])
                                track_predictions += list([y_pred])

                    else:

                        if len(track) >= self.min_hits:

                            self.labels_[track] = track_id
                            self.predictions_[track] = track_predictions

                            track_id += 1

                        break

                if len(track) >= self.min_hits:

                    self.labels_[track] = track_id
                    self.predictions_[track] = track_predictions
                    track_id += 1
