"""
"""
import numpy as np
import GaussianState as gs


class GaussianOperator:
    """
    Class for implementing state-space transformations and obtaining their matrix representations

    ...

    Attributes
    ----------
    basis : str
        Either "pos_mom" or "create_annihilate"

    Methods
    -------
    displace(gs_in: gs.GaussianState, alpha: np.ndarray) -> gs.GaussianState


    """

    def __init__(self, basis: str):
        """
        Parameters
        ----------
        basis : str
            Either "pos_mom" or "create_annihilate" depending on the desired basis

        Raises
        ------
        Exception
            If the basis string is not a valid basis
        """

        if basis not in ("pos_mom", "create_annihilate"):
            raise Exception("Basis not valid")
        self.basis = basis

    def displace(self, gs_in: gs.GaussianState, alpha: np.ndarray) -> gs.GaussianState:
        """Returns the displaced Gaussian state when applying displacement parameter a to it

        Parameters
        ----------
        gs_in : gs.GaussianState
            The input Gaussian state to be displaced
        alpha : np.ndarray
            The displacement parameters

        Returns
        -------
        gs.GaussianState
            The transformed Gaussian state
        """
        n_out = gs_in.n
        x_out = gs_in.get_mu_x() + np.sqrt(2) * np.reshape(np.real(alpha), (-1, 1))
        p_out = gs_in.get_mu_p() + np.sqrt(2) * np.reshape(np.imag(alpha), (-1, 1))

        mu_out = np.empty((x_out.size + p_out.size, 1))
        mu_out[0::2] = x_out
        mu_out[1::2] = p_out

        sigma_out = gs_in.sigma

        return gs.GaussianState(n_out, mu_out, sigma_out)

    def squeeze(self, gs_in: gs.GaussianState, t: np.ndarray) -> gs.GaussianState:
        """Returns the squeezed Gaussian state when applying squeezing (with parameters t)

        Parameters
        ---------
        gs_in : gs.GaussianState
            The input Gaussian state to be squeezed
        t : np.ndarray
            The squeezing parameters

        Returns
        -------
        gs.GaussianState
            The transformed Gaussian state
        """
        S = np.empty((0, 0))

        for r in t:
            s = np.array([[np.exp(-r), 0], [0, np.exp(r)]])
            S = np.block([[S, np.zeros((np.shape(S)[0], 2))],
                                       [np.zeros((2, np.shape(S)[1])), s]])

        n_out = gs_in.n
        mu_out = S @ gs_in.mu
        sigma_out = S @ gs_in.sigma @ S.T

        return gs.GaussianState(n_out, mu_out, sigma_out)

    def interferometer(self, gs_in: gs.GaussianState, U: np.ndarray) -> gs.GaussianState:
        """Returns the Gaussian state transformed by a linear interferometer

        Here U = X+iY encoding S = block([[X, -Y], [Y, X]]) (the symplectic transformation implemented by the
        interferometer)

        Parameters
        ----------
        gs_in : gs.GaussianState
            The input Gaussian state to be squeezed
        U : np.ndarray
            The complex-valued unitary corresponding to the linear interferometer

        Returns
        -------
        gs.GaussianState
            The transformed Gaussian state
        """

        Re_U = np.real(U)
        Im_U = np.imag(U)
        S = np.block([[Re_U, -Im_U], [Im_U, Re_U]])

        n_out = gs_in.n
        mu_out = S @ gs_in.mu
        sigma_out = S @ gs_in.sigma @ S.T

        return gs.GaussianState(n_out, mu_out, sigma_out)
