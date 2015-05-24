import scipy


class grid():
    def __init__(self, num_of_points):
        self.num_of_points = num_of_points;
        self.grid = scipy.polynomial.laguerre.laggauss(num_of_points)
