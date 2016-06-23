__author__ = 'mikhail91'

import numpy
import matplotlib.pyplot as plt

class HoughLinear(object):

    def __init__(self, k_params, b_params, n_candidates, show=False):
        """
        This class is realization of the Hough Transform for the straight tracks in 2D. y = kx + b.
        :param k_params: tuple (min, max, n_bins), bins parameters for the k parameter.
        :param b_params: tuple (min, max, n_bins), bins parameters for the b parameter.
        :param n_candidates: int, number of tracks should be found.
        :param show: boolean, if true, the histograms of the Hough Transform will be shown.
        :return:
        """

        self.k_params = k_params
        self.b_params = b_params
        self.n_candidates = n_candidates
        self.show = show
        self.labels_ = None

    def _hough_transform(self, x, y, k_params, b_params):
        """
        This function is for the Hough Transform of one hit with coordinates (x, y).
        :param x: float: x-coordinate of the hit.
        :param y: float: y-coordinate of the hit.
        :param k_params: tuple (min, max, n_bins), bins parameters for the k parameter.
        :param b_params: tuple (min, max, n_bins), bins parameters for the b parameter.
        :return:
        """

        X_hough = numpy.linspace(k_params[0], k_params[1], 10 * k_params[2])
        Y_hough = -x * X_hough + y

        return X_hough, Y_hough

    def _fit_one(self, X, Y, labels, num_candidate, weights_mul=10.):
        """
        This function makes Hough Transform of all hits and searching for just one track.
        :param X: numpy.array shape = [n_hits], array of x-coordinates of the hits.
        :param Y: numpy.array shape = [n_hits], array of y-coordinates of the hits.
        :param labels: numpy.array shape = [n_hits], labels of the hits. 0 means that this hit was not been used yet.
        :param num_candidate: int, a track candidate number.
        :param weights_mul: float, weight multiplier for the used hits.
        :return: labels: numpy.array shape = [n_hits], labels of the hits. 0 means that this hit was not been used yet.,
                 ind_cand: indeces of the hits in the found track.
        """

        X_hough_all = []
        Y_hough_all = []
        weights_all = []
        ind_all= []

        for ind, (x, y, lab) in enumerate(zip(X, Y, labels)):

            if lab == 0:

                X_hough, Y_hough = self._hough_transform(x, y, self.k_params, self.b_params)
                X_hough_all += list(X_hough.reshape(-1))
                Y_hough_all += list(Y_hough.reshape(-1))
                weights_all += [1.] * len(Y_hough.reshape(-1))
                ind_all += [ind]*len(X_hough.reshape(-1))

            elif lab != 0 and weights_mul != None:

                X_hough, Y_hough = self._hough_transform(x, y, self.k_params, self.b_params)
                X_hough_all += list(X_hough.reshape(-1))
                Y_hough_all += list(Y_hough.reshape(-1))
                weights_all += [1./weights_mul] * len(Y_hough.reshape(-1))
                ind_all += [ind]*len(X_hough.reshape(-1))


        XY_hough = numpy.concatenate((numpy.array(X_hough_all).reshape((-1,1)),
                                      numpy.array(Y_hough_all).reshape((-1,1)),
                                      numpy.array(ind_all).reshape((-1,1))), axis=1)
        weights = numpy.array(weights_all).reshape(-1)



        if self.show==True:
            plt.figure(figsize=(10, 7))
            (counts, xedges, yedges, _) = plt.hist2d(x=XY_hough[:,0], y=XY_hough[:,1], weights=weights,
                                                     range=[[self.k_params[0], self.k_params[1]],
                                                            [self.b_params[0], self.b_params[1]]],
                                                    bins=[self.k_params[2], self.b_params[2]])
            plt.colorbar()

        elif self.show==False:
            (counts, xedges, yedges) = numpy.histogram2d(x=XY_hough[:,0], y=XY_hough[:,1], weights=weights,
                                                         range=[[self.k_params[0], self.k_params[1]],
                                                            [self.b_params[0], self.b_params[1]]],
                                                         bins=[self.k_params[2], self.b_params[2]])

        if self.show:
            plt.show()
        else:
            pass
            # plt.clf()
            # plt.close()


        k_max_ind, b_max_ind = numpy.unravel_index(indices=counts.argmax(), dims=counts.shape)

        k_min, k_max = xedges[k_max_ind:k_max_ind+2]
        b_min, b_max = yedges[b_max_ind:b_max_ind+2]

        sel = (XY_hough[:,0] >= k_min) * (XY_hough[:,0] < k_max) * \
              (XY_hough[:,1] >= b_min) * (XY_hough[:,1] < b_max)
        XY_hough_cand = XY_hough[sel]

        ind_cand = list(numpy.unique(XY_hough_cand[:, 2]))

        labels[ind_cand] = num_candidate

        return labels, ind_cand

    def fit(self, X, Y, weights_mul=10.):
        """
        This function makes Hough Transform of all hits and searching for tracks.
        :param X: numpy.array shape = [n_hits], array of x-coordinates of the hits.
        :param Y: numpy.array shape = [n_hits], array of y-coordinates of the hits.
        :param weights_mul: float, weight multiplier for the used hits.
        :return:
        """

        labels = numpy.zeros(len(X))
        candidates = []

        for num in range(1, self.n_candidates + 1):

            labels, ind_cand = self._fit_one(X, Y, labels, num, weights_mul)
            candidates.append(ind_cand)

        self.labels_ = labels - 1.
