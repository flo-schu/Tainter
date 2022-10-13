from scipy import stats

def f_admin(x, t, N, p_e, epsilon, rho, phi, psi, c, beta, alpha):
    """
    x is the number of administrators
    """
    if x >= N:
        return 0
    else:
        return (
            p_e * (N - 2 * x) +
            stats.beta.cdf(
                epsilon * N / (
                    ((N - x) * (1 - rho) ** x) ** psi +
                    c * ((N - x) * (1 - (1 - rho) ** x)) ** phi
                ),
                a=beta, 
                b=alpha
            )
        )


def f_energy(x, N, rho, phi, psi, c):
    return (
        (((N - x) * (1 - rho) ** x) ** psi + 
        c * ((N - x) * (1 - (1 - rho) ** x)) ** phi) / N
    )

