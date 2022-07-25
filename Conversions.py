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

    p = np.asarray([[1 / np.sqrt(2), 1j / np.sqrt(2)], [1 / np.sqrt(2), -1j / np.sqrt(2)]], dtype="complex_")

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
    p = np.asarray([[1 / np.sqrt(2), 1 / np.sqrt(2)], [-1j / np.sqrt(2), 1j / np.sqrt(2)]], dtype="complex_")

    P = np.empty((0, 0))
    for i in range(n):  # iteratively create block diagonal change of basis matrix
        P = np.block([[P, np.zeros((np.shape(P)[0], 2))],
                      [np.zeros((2, np.shape(P)[1])), p]])

    return P

def symplectic_form_cob():
    """A convenience changing from StrawberryFields' symplectic form, with 2 blocks, to the
    commonly used block-diagonal symplectic form with n blocks
    """
    # TODO Implement this
    return NotImplementedError()

