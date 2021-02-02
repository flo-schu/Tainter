# Tainter

Network model of simple societies inspired by Joseph Tainter's theory of the
collapse of complex societies.

Tainter, J. A. (1988).The collapse of complex societies.  
New studies in archaeology. Cambridge: CambridgeUniversity Press

## Model

The model is an agent-based network model written in Python. The main files
are included under scripts/model.

## Instructions

to execute the model code, navigate (cd) into scripts directory. From here all
scripts in the main directory can be executed.

The files __macroscopic_approximation.py__ and __stochastic_model.py__ are
very basic scripts which execute the main modules of the code.

In order to modify the output plot of __stochastic_plot.py__,
__stochastic_model.py__ needs to be executed with desired paramter settings.
The corresponding directories will appear under /data/YYYYMMDD_HHMM.
Where the time corresponds to the current system time.
These names need to be inserted into __stochastic_plot.py__. As a help,
three example runs have been already included in a data directory.

## Analyses

Several analyses can be performed with the model code.

+ stochastic simulations of the network with various paramter settings.
    For details on parameters see the inline documentation in model/main.py

+ analytic calculations with equations derived in publication
