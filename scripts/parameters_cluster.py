import numpy as np
import sys
from tainter.cluster.parameter_setup import generate_parameters

# select parameter ranges
rho = np.linspace(0, 0.3, 100)  # link density in erdos renyi network
c = np.linspace(1, 3, 200)  # efficiency of coordinated Workers
pe_null = np.array([0])
pe_explore = np.logspace(-5, -1.6, num=499)
p_e = np.concatenate((pe_null, pe_explore), axis=None)

approx_parameters = dict(
    N=400,
    p_e=None,
    rho=None,
    psi=0.75,
    phi=0.75,
    c=None,
    alpha=1,
    beta=15
)


parameters, n_chunks = generate_parameters(
    output=sys.argv[1],
    approx_parameters=approx_parameters,
    chunk_size=1000,
    p_e=p_e,
    rho=rho,
    c=c,
)

print(f"prepared parameters {parameters} in {n_chunks} chunks.")