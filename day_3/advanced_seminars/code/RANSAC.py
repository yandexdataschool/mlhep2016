__author__ = 'mikhail91'

class RANSAC(object):

    def __init__(self, sklearn_ransac):
        """
        This class is wrapper over the scikit-learn RANSACRegressor.
        :param sklearn_ransac: a scikit-learn RANSACRegressor object.
        :return:
        """

        self.sklearn_ransac = sklearn_ransac
        self.labels_ = None

    def fit(self, x, y):
        """
        Fit the method.
        :param x: numpy.ndarray shape=[n_hits], X of hits.
        :param y: numpy.array shape=[n_hits], y of hits.
        :return:
        """

        self.sklearn_ransac.fit(x.reshape(-1,1), y)

        self.labels_ = self.sklearn_ransac.inlier_mask_ - 1.