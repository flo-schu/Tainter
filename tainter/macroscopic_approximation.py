import numpy as np
import scipy.stats as sci
from scipy.integrate import odeint
from scipy.integrate import trapz
import matplotlib.pyplot as plt

# Parameters ------------------------------------------------------------------
N = 400  # Network size
epsilon = 1  # threshold
beta = 15  # scale parameter of beta distribution
alpha = 1  # location parameter of beta distribution
p_e = 0.0027 # exploration
rho = 0.002  # link density
phi = 1.333  # efficiency
t = np.linspace(0, 10000, 10001) # timesteps of solver

# Definition Equations --------------------------------------------------------
def f_a(x, t, N, p_e, epsilon, rho, phi, beta, alpha):
    if x >= N:
        return 0
    else:
        return (
                p_e * (N - 2 * x) +
                sci.beta.cdf(
                    (((epsilon * N) / (((N - x) * (1 - rho) ** x) +
                                       ((N - x) * (1 - (1 - rho) ** x)) ** phi))),
                    a=beta, b=alpha)
        )


def f_e(x, N, rho, phi):
    return (
            ((N - x) * (1 - rho) ** x + ((N - x) * (1 - (1 - rho) ** x)) ** phi) / N
    )


def get_st(t, e):
    if all(np.round(e, 6) > 0):
        return int(np.max(t))
    else:
        return int(np.where(np.round(e, 6) == 0)[0][0]+1)


# Calculation ------------------------------------------------------------------

result = odeint(f_a, y0=0, t=t, args=(N, p_e, epsilon, rho, phi, beta, alpha))
result[result > N] = N  # turn all x > N to N (fix numerical issue)
e = f_e(result[:, 0], N, rho, phi)
st = get_st(t, e)
te = trapz(e[:st], t[:st])

plt.cla()
plt.plot(t, result/N, label="admin")
plt.plot(t, e, label="energy")
plt.xlabel("time")
plt.ylabel("administration share / energy cap-1")
plt.legend()
plt.xscale('log')
plt.ylim(0,1.2)
plt.show()
