# Tainter

Network model of simple societies inspired by Joseph Tainter's theory of the
collapse of complex societies.

Tainter, J. A. (1988).The collapse of complex societies.  
New studies in archaeology. Cambridge: CambridgeUniversity Press

## installation

create environment, activate it and install model package

```bash
conda create -n tainter
conda activate tainter
conda install python=3.9
pip install -e .
```

## Model

The model is an agent-based network model written in Python. The main files
are included under scripts/model.

## Instructions

to execute the model code, navigate (cd) into scripts directory. From here all
scripts in the main directory can be executed.

Prefixes correspond to figures displayed in the publication

The files __macroscopic_approximation.py__ and __stochastic_model.py__ are
very basic scripts which execute the main modules of the code.

data for f4 was generated on a high perfomance cluster. The necessary files for
this computation are found under scripts/cluster. These files need to be slightly
adapted to suit file structure. Simulations can be carried out on a normal
computer if the resolution of parameters is reduced.

## Analyses

Several analyses can be performed with the model code.

+ stochastic simulations of the network with various paramter settings.
    For details on parameters see the inline documentation in model/main.py

+ analytic calculations with equations derived in publication

+ several plots to illustrate the results in publication