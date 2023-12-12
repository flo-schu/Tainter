import numpy as np
from scipy import stats
from scipy.integrate import odeint, trapz

def f_admin(x, t, N, p_e, rho, phi, psi, c, alpha, beta):
    """
    x is the number of administrators.

    Note that the switch: b=alpha and a=beta are is done on purpose. This is no error, 
    it is motivated in the manuscript:
    
    As $1 - B \sim \mathrm{Beta}(\beta, \alpha)$. Hence the probability of 
    $E/N < \epsilon$ is given by the cumulative probability function of the 
    Beta distribution, $P = F(N^a \epsilon / e, \beta, \alpha)$.

    I.e. it is necessary to compute the CDF of the Complementary Beta distribution.
    Therefore a and b are mapped to beta and alpha, respectively

    Note also that 
    phi := a
    psi := b
    """
    if x >= N:
        return N
    else:
        return (
            p_e * (N - 2 * x) +
            stats.beta.cdf(
                N ** psi / (
                    ((N - x) * (1 - rho) ** x) ** psi +
                    c * ((N - x) * (1 - (1 - rho) ** x)) ** phi
                ),
                a=beta, 
                b=alpha
            )
        )


def f_energy(x, N, rho, phi, psi, c):
    # initial access of resources depends on psi in order
    # to satisfy, that the energy per capita production is 1
    return (
        (((N - x) * (1 - rho) ** x) ** psi + 
        c * ((N - x) * (1 - (1 - rho) ** x)) ** phi) / N ** psi
    )


def get_st(t, e):
    if all(np.round(e, 6) > 0):
        return int(np.max(t))
    else:
        return int(np.where(np.round(e, 6) == 0)[0][0] + 1)


def integrate_fa(t, params):
    N = params["N"]
    result = odeint(f_admin, y0=0, t=t, args=tuple(params.values())).flatten()
    result[result > N] = N  # turn all x > N to N (fix numerical issue)
    e = f_energy(result, N, params["rho"], params["phi"], params["psi"], params["c"])
    st = get_st(t, e)
    te = trapz(e[:st], t[:st])
    return st, te, result, e
