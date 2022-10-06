from scipy import stats

# approximation from first submission
def f_a_1(x, t, N, p_e, epsilon, rho, phi, beta, alpha):
    if x >= N:
        return 0
    else:
        return (
                p_e * (N - 2 * x) +
                stats.beta.cdf(
                    (((epsilon * N) / (((N - x) * (1 - rho) ** x) +
                                       ((N - x) * (1 - (1 - rho) ** x)) ** phi))),
                    a=beta, b=alpha)
        )


def f_e_1(x, N, rho, phi):
    return (
            ((N - x) * (1 - rho) ** x + ((N - x) * (1 - (1 - rho) ** x)) ** phi) / N
    )

# updated approximation from revision
def f_a_2(x, t, N, p_e, epsilon, rho, phi, psi, c, beta, alpha):
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


def f_e_2(x, N, rho, phi, psi, c):
    return (
        (((N - x) * (1 - rho) ** x) ** psi + 
        c * ((N - x) * (1 - (1 - rho) ** x)) ** phi) / N
    )

