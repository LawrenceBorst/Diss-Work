"""
n-mode Gaussian states in state space can be defined by Gaussian Wigner functions over an n-dimensional x-p space,
where the position is encoded as r=(x_{0}, p_{1},...,x_{n}, p_{n}).
Gaussian states are therefore fully defined by their zeroth and first moment (mean and covariance)
"""
import numpy as np
import Conversions as conv

H = 2

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

    def __init__(self, n: int, mu: np.ndarray = np.array([]), sigma: np.ndarray = np.array([]), basis="nBlockXP"):
        """
        Parameters
        ----------
        n : int
            The number of modes in the state
        mu : np.ndarray
            The zeroth moment/mean of the Gaussian Wigner function
        sigma : np.ndarray
            THe first moment/covariance of the Gaussian Wigner function
        basis : str
            One of "nBlockXP", "2BlockXP", "nBlockCA" or "2BlockCA" depending on (1) whether we use
            position-momentum or annihilation-creation basis, and (2) the symplectic form

        Raises
        ------
        Exception
            If the basis is not a valid choice
            If the number of modes on the mean/covariance do not match the specified number
        """
        if basis not in ["nBlockXP", "2BlockXP", "nBlockCA", "2BlockCA"]:
            raise Exception("Basis not valid - expected basis written as nBlockXP, 2BlockXP, nBlockCA, or 2BlockCA")

        self.n = n

        if mu.size == 0 or sigma.size == 0:     # If mu or sigma is not specified
            self.mu = np.zeros((2 * n, 1))
            self.sigma = np.identity(2 * n)
            self.basis = basis
        else:
            self.mu = mu
            self.sigma = sigma
            self.basis = basis
            if np.shape(sigma) != (2 * n, 2 * n) or np.shape(mu) != (2 * n, 1):
                raise Exception("Dimensions don't match")

        if self.basis == "nBlockXP":
            return
        elif self.basis == "2BlockXP":
            self.mu = conv.symp_form_n_to_2(self.n) @ self.mu
            self.sigma = conv.symp_form_n_to_2(self.n) @ self.sigma @ conv.symp_form_n_to_2(self.n).T
        elif self.basis == "nBlockCA":
            self.mu = conv.xp_to_ca(self.n) @ self.mu
            self.sigma = conv.xp_to_ca(self.n) @ self.sigma @ conv.xp_to_ca(self.n).T
        elif self.basis == "2BlockCA":
            self.mu = conv.symp_form_n_to_2(self.n) @ conv.xp_to_ca(self.n) @ self.mu
            self.sigma = conv.symp_form_n_to_2(self.n) @ conv.xp_to_ca(self.n) @ self.sigma @ \
                         conv.xp_to_ca(self.n).T @ conv.symp_form_n_to_2(self.n).T


    def get_mu_x(self) -> np.ndarray:
        """Returns the position of the zeroth moment/mean

        Returns
        -------
        The position vector of the mean
        """
        if self.basis == "nBlockXP":
            return self.mu[0::2]
        elif self.basis == "2BlockXP":
            return conv.symp_form_n_to_2(self.n) @ self.mu[0::2]
        elif self.basis == "nBlockCA":
            return conv.xp_to_ca(self.n) @ self.mu[0::2]
        elif self.basis == "2BlockCA":
            return conv.symp_form_n_to_2(self.n) @ conv.xp_to_ca(self.n) @ self.mu[0::2]


    def get_mu_p(self) -> np.ndarray:
        """Returns the momentum of the first moment/covariance

        Returns
        -------
        The momentum vector of the mean
        """
        if self.basis == "nBlockXP":
            return self.mu[1::2]
        elif self.basis == "2BlockXP":
            return conv.symp_form_n_to_2(self.n) @ self.mu[1::2]
        elif self.basis == "nBlockCA":
            return conv.xp_to_ca(self.n) @ self.mu[1::2]
        elif self.basis == "2BlockCA":
            return conv.symp_form_n_to_2(self.n) @ conv.xp_to_ca(self.n) @ self.mu[1::2]
