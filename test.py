import Conversions
import GaussianOperator
import GaussianState
import strawberryfields as sf
import Conversions as conv
import numpy as np

formic = sf.apps.data.Formic()
w = formic.w
wp = formic.wp
Ud = formic.Ud
delta = formic.delta
T = 0
t, U1, r, U2, alpha = sf.apps.vibronic.gbs_params(w, wp, Ud, delta, T)  # Circuit parameters


go = GaussianOperator.GaussianOperator()

S = Conversions.get_symp_form(7)

gs = GaussianState.GaussianState(n=7, basis="2BlockXP")
np.set_printoptions(precision=1)
gs = go.interferometer(gs, U1)  # this function ignores the basis
gs = go.squeeze(gs, r)      # 2 * r as SF uses a different convention for the exponent TODO KEEP IN MIND
ut1, st1, vt1 = sf.decompositions.bloch_messiah(gs.sigma)
gs = go.interferometer(gs, U2)
ut1, st1, vt1 = sf.decompositions.bloch_messiah(gs.sigma)
gs = go.displace(gs, np.sqrt(2) * alpha)     # also basis-independent. Also, SF uses sqrt(2h), with h=2

print(Ud)
