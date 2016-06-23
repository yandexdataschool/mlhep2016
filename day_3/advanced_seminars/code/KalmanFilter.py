__author__ = 'mikhail91'

from copy import copy
from numpy.linalg import inv
import numpy
import pandas

from sklearn.linear_model import LinearRegression

class LinearKalmanFilter(object):

    def __init__(self,
                 window,
                 min_hits=4,
                 initial_state_covariance=numpy.eye(2),
                 transition_covariance=numpy.eye(2),
                 observation_covariance=numpy.eye(1)):
        """
        This is simple realization of the Kalman Filter model for the straight track in 2D.
        :param window: float, if the difference between a hit and the extrapolation function is larger than thi value,
        the hit is rejected.
        :param min_hits: int, min number of hits in a track.
        :param initial_state_covariance: numpy.matrix shape = [2,2], init state covariance.
        :param transition_covariance: numpy.matrix shape = [2,2], transition covariance.
        :param observation_covariance: numpy.matrix shape = [1,1], observation covariance.
        :return:
        """

        self.window = window
        self.min_hits = min_hits
        self.initial_state_covariance = initial_state_covariance
        self.observation_covariance = observation_covariance
        self.transition_covariance = transition_covariance
        self.labels_ = None
        self.predictions_ = None
        self.matrices_ = None

    def fit(self, x, y):
        """
        This function runs the model.
        :param x: numpy.array shape = [n_hits], array of x-coordinates of the hits.
        :param y: numpy.array shape = [n_hits], array of y-coordinates of the hits.
        :return:
        """

        self.labels_ = -1 * numpy.ones(len(x))
        self.predictions_ = copy(y)
        self.matrices_ = []

        layers_x = numpy.unique(x)
        layers = numpy.arange(len(layers_x))
        hits_indeces = numpy.arange(len(x))

        track = []
        track_id = 0

        # Loop over all seeds
        for first_i in hits_indeces[x == layers_x[0]]:
            for second_i in hits_indeces[x == layers_x[1]]:

                if self.labels_[first_i] != -1 or self.labels_[second_i] != -1:
                    continue

                track = [first_i, second_i]
                track_predictions = [y[first_i], y[second_i]]



                #y = kx + b
                k = 1. * (y[second_i] - y[first_i])/((x[second_i] - x[first_i]))
                b = y[first_i] - k * x[first_i]

                # Init the Kalman filter
                state = numpy.matrix([[k], [b]])
                P = self.initial_state_covariance
                Q = self.transition_covariance
                R = self.observation_covariance
                I = numpy.eye(2)
                F = numpy.eye(2)

                state_series = []
                P_series = []
                prediction_series = []
                residual_series = []
                S_series = []
                K_series = []

                # Loop over all layers
                for layer_id in layers[2:]:

                    layer_x = layers_x[layer_id]



                    state = F * state
                    P = F * P * F.T + Q

                    H = numpy.matrix([[layer_x, 1]])
                    y_predicted = H * state




                    hit_index_min_y_residual = -1
                    y_residual_min = numpy.matrix([[self.window]])

                    ys_in_layer = y[x == layers_x[layer_id]]
                    hits_indeces_in_layer = hits_indeces[x == layers_x[layer_id]]

                    for hit_index in hits_indeces_in_layer:

                        y_observed = numpy.matrix([[y[hit_index]]])
                        y_residual = y_observed - y_predicted


                        if numpy.abs(y_residual)[0,0] <= numpy.abs(y_residual_min)[0,0]:

                            y_residual_min = y_residual
                            hit_index_min_y_residual = hit_index





                    if hit_index_min_y_residual != -1:

                        track.append(hit_index_min_y_residual)
                        track_predictions.append(y_predicted[0,0])

                    elif len(track) >= self.min_hits:

                        self.labels_[track] = track_id
                        self.predictions_[track] = track_predictions
                        track_id += 1
                        metrices_dict = {'state': state_series,
                                         'P':P_series,
                                         'prediction':prediction_series,
                                         'residual': residual_series,
                                         'S':S_series,
                                         'K':K_series}
                        self.matrices_ += [metrices_dict]
                        break




                    S = H * P * H.T + R
                    K = P * H.T * inv(S)

                    state = state + K * y_residual_min
                    P = (I - K * H) * P

                    state_series += [state]
                    P_series += [P]
                    prediction_series += [y_predicted]
                    residual_series += [y_residual_min]
                    S_series += [S]
                    K_series += [K]




                if len(track) >= self.min_hits:

                    self.labels_[track] = track_id
                    self.predictions_[track] = track_predictions
                    track_id += 1
                    metrices_dict = {'state': state_series,
                                     'P':P_series,
                                     'prediction':prediction_series,
                                     'residual': residual_series,
                                     'S':S_series,
                                     'K':K_series}
                    self.matrices_ += [metrices_dict]



