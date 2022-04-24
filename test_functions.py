import numpy as np
# This function was copied from: https://github.com/oxfordcontrol/Bayesian-Optimization/blob/master/benchmark_functions.py

class Hart6:
    '''
    Hartmann 6-Dimensional function
    Based on the following MATLAB code:
    https://www.sfu.ca/~ssurjano/hart6.html
    '''
    def __init__(self, sd=0):
        self.sd = sd
        self.bounds = np.array([[0, 1], [0, 1], [0, 1],
                               [0, 1], [0, 1], [0, 1]])
        self.min = np.array([0.20169, 0.150011, 0.476874,
                             0.275332, 0.311652, 0.6573])
        self.fmin = -3.32237

    def f(self, xx):
        if len(xx.shape) == 1:
            xx = xx.reshape((1, 6))

        assert xx.shape[1] == 6

        n = xx.shape[0]
        y = np.zeros(n)
        for i in range(n):
            alpha = np.array([1.0, 1.2, 3.0, 3.2])
            A = np.array([[10, 3, 17, 3.5, 1.7, 8],
                          [0.05, 10, 17, 0.1, 8, 14],
                          [3, 3.5, 1.7, 10, 17, 8],
                          [17, 8, 0.05, 10, 0.1, 14]])
            P = 1e-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                                 [2329, 4135, 8307, 3736, 1004, 9991],
                                 [2348, 1451, 3522, 2883, 3047, 6650],
                                 [4047, 8828, 8732, 5743, 1091, 381]])

            outer = 0
            for ii in range(4):
                inner = 0
                for jj in range(6):
                    xj = xx[i, jj]
                    Aij = A[ii, jj]
                    Pij = P[ii, jj]
                    inner = inner + Aij*(xj-Pij)**2

                new = alpha[ii] * np.exp(-inner)
                outer = outer + new

            y[i] = -outer

        if self.sd == 0:
            noise = np.zeros(n)
        else:
            noise = np.random.normal(0, self.sd, n)

        return (y + noise).reshape((n, 1))