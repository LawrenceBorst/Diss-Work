"""
n-mode Gaussian states in state space can be defined by Gaussian Wigner functions over an n-dimensional x-p space,
where the position is encoded as r=(x_{0}, p_{1},...,x_{n}, p_{n}).
Gaussian states are therefore fully defined by their zeroth and first moment (mean and covariance)
"""
import numpy as np


class GaussianState:
    """
    Implements Gaussian states in position-momentum (state) space
    ...

    Attributes
    ----------
    n : int
        (half the) dimension of the state space
    mu : np_arr
        mean state vector
    sigma : np_arr
        covariance of the state vector

    Methods
    -------
    get_mu_x() -> np.ndarray
        Returns the position of the zeroth moment/mean
    get_mu_p() -> np.ndarray
        Returns the moomentum of the first moment/covariance
    """

    def __init__(self, n: int, mu: np.ndarray = np.array([]), sigma: np.ndarray = np.array([])):    # TODO add basis
        """
                Parameters
                ----------
                n : int
                    The number of modes in the state
                mu : np.ndarray
                    The zeroth moment/mean of the Gaussian Wigner function
                sigma : np.ndarray
                    THe first moment/covariance of the Gaussian Wigner function

                Raises
                ------
                Exception
                    If the number of modes on the mean/covariance do not match the specified number
        """

        self.n = n

        if mu.size == 0 or sigma.size == 0:
            self.mu = np.zeros((2 * n, 1))
            self.sigma = np.identity(2 * n)
        else:
            self.mu = mu
            self.sigma = sigma
            if np.shape(sigma) != (2 * n, 2 * n) or np.shape(mu) != (2 * n, 1):
                raise Exception("Dimensions don't match")

    def get_mu_x(self) -> np.ndarray:   # TODO add basis
        """Returns the position of the zeroth moment/mean
        """
        return self.mu[0::2]

    def get_mu_p(self) -> np.ndarray:   # TODO add basis
        """Returns the momentum of the first moment/covariance
        """
        return self.mu[1::2]