# created by Florian Schunck on 26.06.2020
# Project: tainter
# Short description of the feature:
# 1. create a parameter grid and save it to chunks of size n. The last chunk
#    contains the tailing elements and can be smaller than n.
# 2. allows to create one chunk for each parameter combination or one chunk
#    with all combination or any number in between
#
# ------------------------------------------------------------------------------
# Open tasks:
# TODO:
#
# ------------------------------------------------------------------------------
import os
import json
import itertools as it
import numpy as np
from tqdm import tqdm

def generate_parameters(output, approx_parameters, chunk_size=None, **kwargs):
    parameters = list(kwargs.keys())

    os.makedirs(output, exist_ok=True)

    with open(os.path.join(output, "approx_params.json"), "w") as f:
        json.dump(approx_parameters, f)

    
    # parameter grid
    pargrid = np.array(list(it.product(*kwargs.values())))
    n_pargrid = len(pargrid)

    # choose n_pargrid if only one chunk should be computed,
    # choose 1 if n chunks should be created
    if chunk_size is None:
        chunk_size = n_pargrid

    n_chunks = int(np.ceil(n_pargrid / chunk_size))


    print("creating parameter chunks")
    with tqdm(total=n_chunks) as pbar:
        # save paramter chunks
        for chunk in range(n_chunks):
            lower_slice = chunk * chunk_size
            upper_slice = min((chunk + 1) * chunk_size, n_pargrid)
            data_chunk = pargrid[lower_slice:upper_slice]
            np.savetxt(
                os.path.join(output, f"params_{str(chunk+1).zfill(4)}.txt"),
                data_chunk,
                delimiter=",", newline="\n"
            )
            pbar.update(1)

    return parameters, n_chunks