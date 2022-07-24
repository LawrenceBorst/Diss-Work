import GaussianOperator
import GaussianState
import strawberryfields as sf

formic = sf.apps.data.Formic()
w = formic.w
wp = formic.wp
Ud = formic.Ud
delta = formic.delta
T = 0
t, U1, r, U2, alpha = sf.apps.vibronic.gbs_params(w, wp, Ud, delta, T)  # Circuit parameters

go = GaussianOperator.GaussianOperator("pos_mom")
gs = GaussianState.GaussianState(7)
gs = go.interferometer(gs, U1)
gs = go.squeeze(gs, r)
gs = go.interferometer(gs, U2)
gs = go.displace(gs, alpha)

print(gs.mu)
print(gs.sigma)