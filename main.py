'''
Script deriving the Frank Condon Profile(s) (FCP) to the case with a zero-temperature vacuum input state
'''

import numpy as np
import strawberryfields as sf
import matplotlib.pyplot as plt

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

# custom inverse fourier transform because the numpy one might get the exponent wrong (will compare later)
# TODO turn it into a FAST Fourier Transform later
def IFT(w_vib, Gk_arr, K):
    theta = 2 * np.pi / (K + 1)
    exp_arr = np.exp(1j * theta * w_vib * np.arange(0, K + 1, 1, dtype="complex_"))
    return 1 / (K + 1) * np.dot(Gk_arr, exp_arr)

def get_Fourier_k(k, K, M, U, delta):
    r"""Gets the kth Fourier component of the FCP, where k is the wavenumber

    Args:
        k (int): index of the Fourier component
        K (int): total number of bins of the discretized distribution
        m (int): total number of modes
        U (array): The Doktorov operation accounting for the linear optical circuit
    """
    gamma_hat = np.exp(2 * r) - 1
    N = np.prod(np.sqrt(1 + gamma_hat) / (np.pi * gamma_hat))  # N

    Gamma = np.diagflat(gamma_hat)  # Gamma

    theta = 2 * np.pi / (K + 1)
    phi = -k * theta * w
    Phi = np.diagflat(np.exp(1j * phi) - 1)  # Phi

    a = U.T @ Phi @ delta.conj() / np.sqrt(2)       # a
    b = U.conj().T @ Phi @ delta / np.sqrt(2)       # b
    c = np.hstack((a, b))                           # c
    c0 = delta.T @ Phi @ delta.conj() / np.sqrt(2)  # c0

    diag = np.diagflat(np.exp(1j * phi))
    Q00 = 2 * np.linalg.inv(Gamma) + np.identity(M)
    Q01 = - U.T @ diag @ np.conjugate(U)
    Q10 = - np.conjugate(U).T @ diag @ U
    Q = np.block([
        [Q00, Q01],
        [Q10, Q00]
    ])  # Q

    G_tilde_k = N * np.power(2 * np.pi, M) / \
                np.sqrt(np.linalg.det(Q)) * \
                np.exp(1 / 2 * c.T @ np.linalg.inv(Q) @ c + c0)

    return G_tilde_k

# Quick test with receiving the Fourier components
K = 1000  # resolution
max = 8000  # maximum

# x
x = np.arange(0, max, max / (K + 1))
y_hat = np.empty(K + 1, dtype="complex_")

for i in range(K + 1):
    k = i * max / (K + 1)
    y_hat[i] = get_Fourier_k(k, K, len(U2), U2, delta)  # TODO The matrix we want is from the Bloch-Messiah decomposition
                                                    # TODO for the Doktorov transformation. NOT U2, because SF
                                                    # TODO employs a slightly different decomposition

# Quick test, seeing if the FCPs are actually real numbers  # TODO check if the prefactor 1/(K+1) is accounted for
#plt.plot(x, np.real(np.fft.ifft(y)))

ift = np.vectorize(IFT)

y = np.empty(K + 1, dtype="complex_")
for i in range(len(y_hat)):
    y[i] = np.real(IFT(i * max / (K + 1), y_hat, K))

plt.plot(x, y)

###########
# SF CODE #
###########
nr_samples = 500
s = sf.apps.qchem.vibronic.sample(t, U1, r, U2, alpha, nr_samples)
e = sf.apps.qchem.vibronic.energies(s, w, wp)

#full_spectrum = sf.apps.plot.spectrum(e, xmin=0, xmax=max)
#full_spectrum.show()

plt.show()
