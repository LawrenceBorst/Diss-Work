"""
Change of basis matrices
Change from x-p space giving (x_{0}, p_{0},..., x_{n}, p_{n})
to
(a_{0}, c_{0},...,a_{n}, c_{n})

OR

from x-p space with the symplectic form giving (x_{0}, p_{0},..., x_{n}, p_{n})
to (x_{0},..., x_{n}, p_{0},..., p_{n})
"""

import numpy as np

H = 2


def get_symp_form(n: int) -> np.ndarray:
    return np.block([[np.zeros((n, n)), np.identity(n)],
                     [-np.identity(n), np.zeros((n, n))]])


def xp_to_ca(n: int) -> np.ndarray:
    """Returns a conversion matrix from x-p space (r = (x_{0}, p_{0},..., x_{n}, p_{n}))
    to c-a space (r = (a_{0}, c_{0},..., a_{n}, c_{n}))

    Parameters
    ----------
    n : int
        Number of modes

    Returns
    -------
    The 2n x 2n change of basis matrix
    """

    p = np.asarray([[np.sqrt(H) / np.sqrt(2), np.sqrt(H) * 1j / np.sqrt(2)],
                    [-1j * np.sqrt(H) / np.sqrt(2), 1j * np.sqrt(H) / np.sqrt(2)]], dtype="complex_")

    P = np.empty((0, 0))
    for i in range(n):  # iteratively create block diagonal change of basis matrix
        P = np.block([[P, np.zeros((np.shape(P)[0], 2))],
                      [np.zeros((2, np.shape(P)[1])), p]])

    return P


def ca_to_xp(n: int) -> np.ndarray:
    """Returns a conversion matrix from c-a space (r = (a_{0}, c_{0},..., a_{n}, c_{n}))
    to x-p space (r = (x_{0}, p_{0},..., x_{n}, p_{n}))

    Parameters
    ----------
    n : int
        Number of modes

    Returns
    -------
    The 2n x 2n change of basis matrix
    """
    p = np.asarray([[1 / np.sqrt(2 * H), 1j / np.sqrt(2 * H)], [1 / np.sqrt(2 * H), -1j / np.sqrt(2 * H)]], dtype="complex_")

    P = np.empty((0, 0))
    for i in range(n):  # iteratively create block diagonal change of basis matrix
        P = np.block([[P, np.zeros((np.shape(P)[0], 2))],
                      [np.zeros((2, np.shape(P)[1])), p]])

    return P

def symp_form_2_to_n(n : int) -> np.ndarray:
    """A convenience changing from StrawberryFields' symplectic form, with 2 blocks, to the
    commonly used block-diagonal symplectic form with n blocks

    Parameters
    ----------
    n : int
        Number of modes

    Returns
    -------
    The 2n x 2n change of basis matrix
    """
    P = np.zeros((2*n, 2*n))
    for i in range(n):
        P[2*i][i] = 1
        P[2*i+1][n+i] = 1

    return P


def symp_form_n_to_2(n : int) -> np.ndarray:
    """A convenience changing from StrawberryFields' symplectic form, with n blocks, to the
    commonly used block-diagonal symplectic form with 2 blocks

    Parameters
    ----------
    n : int
        Number of modes

    Returns
    -------
    The 2n x 2n change of basis matrix
    """
    P = np.zeros((2*n, 2*n))
    for i in range(n):
        P[i][2*i] = 1
        P[n+i][2*i+1] = 1

    return P


