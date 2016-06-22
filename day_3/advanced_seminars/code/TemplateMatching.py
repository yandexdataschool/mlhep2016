__author__ = 'mikhail91'

import numpy
import pandas
from sklearn.linear_model import LinearRegression

class SimpleTemplateMatching(object):

    def __init__(self, n_hits, window_width):
        """
        This class is simple realization of a Template Matching paradigm for straight tracks in 2D.
        :param n_hits: int, min number of hits to consider the track recognized.
        :param window_width: float, width of a searching window for searching hits for a track.
        :return:
        """

        self.window_width = window_width
        self.n_hits = n_hits

    def fit(self, x, y):
        """
        Fit the method.
        :param x: numpy.ndarray shape=[n_hits, n_features], X of hits.
        :param y: numpy.array shape=[n_hits], y of hits.
        :return:
        """

        used = numpy.zeros(len(x))
        labels = -1. * numpy.ones(len(x))
        track_id = 0
        tracks_params = []

        for first_ind in range(len(x)):

            for second_ind in range(len(x)):

                x1 = x[first_ind]
                y1 = y[first_ind]

                x2 = x[second_ind]
                y2 = y[second_ind]

                if (x1 >= x2) or (used[first_ind] == 1) or (used[second_ind] == 1):
                    continue

                k = 1. * (y2 - y1) / (x2 - x1)
                b = y1 - k * x1

                y_upper = b + k * x.reshape(-1) + self.window_width
                y_lower = b + k * x.reshape(-1) - self.window_width

                track = (y <= y_upper) * (y >= y_lower) * (used == 0)

                if track.sum() >= self.n_hits:

                    used[track] = 1
                    labels[track] = track_id
                    track_id += 1

                    X_track = x[track]
                    y_track = y[track]

                    lr = LinearRegression()
                    lr.fit(X_track.reshape(-1,1), y_track)

                    params = list(lr.coef_) + [lr.intercept_]
                    tracks_params.append(params)


        self.labels_ = labels
        self.tracks_params_ = numpy.array(tracks_params)


from sklearn.linear_model import RANSACRegressor
from copy import copy

class RANSACTemplateMatching(object):

    def __init__(self, n_hits, ransac_estimator):
        """
        This class is an implementation of the RANSAC algorithm to the tracks recognition.
        :param n_hits: int, min number of hits to consider the track recognized.
        :param ransac_estimator: object, RANSAC estimator.
        :return:
        """

        self.n_hits = n_hits
        self.ransac_estimator = ransac_estimator

    def fit(self, x, y):
        """
        Fit the method.
        :param x: numpy.ndarray shape=[n_hits], X of hits.
        :param y: numpy.array shape=[n_hits], y of hits.
        :return:
        """

        used = numpy.zeros(len(x))
        labels = -1. * numpy.ones(len(x))
        indeces = numpy.array(range(len(x)))
        track_id = 0
        tracks_params = []

        flag = 1
        while flag == 1 and len(x) >= self.n_hits:

            estimator = copy(self.ransac_estimator)

            estimator.fit(x.reshape(-1,1), y)
            mask = estimator.inlier_mask_

            if mask.sum() >= self.n_hits:

                labels[indeces[mask]] = track_id
                track_id += 1

                best = estimator.estimator_

                params = list(best.coef_) + [best.intercept_]
                tracks_params.append(params)

                x = x[mask == 0]
                y = y[mask == 0]
                indeces = indeces[mask == 0]

            else:

                flag = 0

        self.labels_ = labels
        self.tracks_params_ = numpy.array(tracks_params)
