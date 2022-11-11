import numpy as np
from scipy import stats
from scipy.integrate import odeint, trapz

def f_admin(x, t, N, p_e, rho, phi, psi, c, beta, alpha):
    """
    x is the number of administrators
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
                b=beta,
                a=alpha 
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
