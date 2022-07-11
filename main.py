'''
Script deriving the Frank Condon Profile(s) (FCP) to the case with a zero-temperature vacuum input state
'''

import numpy as np
import strawberryfields as sf

############
# GET DATA #
############

formic = sf.apps.data.Formic()
w = formic.w
wp = formic.wp
Ud = formic.Ud
delta = formic.delta
T = 0
t, U1, r, U2, alpha = sf.apps.vibronic.gbs_params(w, wp, Ud, delta, T)  # Circuit parameters

def get_Fourier_k(k, K, M, U):
    r"""Gets the kth Fourier component of the FCP, where k is the wavenumber

    Args:
        k (int): index of the Fourier component
        K (int): total number of bins of the discretized distribution
        m (int): total number of modes
        U (array): The Doktorov operation accounting for the linear optical circuit
    """
    # TODO check for semantic errors

    gamma_hat = np.exp(2 * r) - 1
    N = np.prod(np.sqrt(1 + gamma_hat) / (np.pi * gamma_hat))   # N

    Gamma = np.diagflat(gamma_hat)                  # Gamma

    theta = 2 * np.pi / (K + 1)
    phi = -k * theta * w
    Phi = np.diagflat(np.exp(1j * phi) - 1)         # Phi

    a = U.T @ Phi @ delta.conj() / np.sqrt(2)       # a
    b = U.conj().T @ Phi @ delta / np.sqrt(2)       # b
    c = np.hstack((a, b))                           # c
    c0 = delta.T @ Phi @ delta.conj() / np.sqrt(2)  # c0

    diag = np.diagflat(np.exp(1j * phi) - 1)
    Q00 = 2 * np.linalg.inv(Gamma) + np.identity(M)
    Q01 = - U.T @ diag @ np.conjugate(U)
    Q10 = - np.conjugate(U).T @ diag @ U
    Q = np.block([[Q00, Q01], [Q10, Q00]])          # Q

    G_tilde_k = N * np.power(2 * np.pi, M) / \
         np.sqrt(np.linalg.det(Q)) * \
         np.exp(1 / 2 * c.T @ np.linalg.inv(Q) @ c + c0)

    return G_tilde_k

# Quick test with receiving the Fourier components
K = 10
components = np.empty(K, dtype="complex")
for k in range(K):
    components[k] = get_Fourier_k(k, K, len(Ud), U1)

# Quick test, seeing if the FCPs are actually real numbers  # TODO check if the prefactor 1/(K+1) is accounted for
print(np.fft.ifft(components))
