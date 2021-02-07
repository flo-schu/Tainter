# Tainter

Network model of simple societies inspired by Joseph Tainter's theory of the
collapse of complex societies.

Tainter, J. A. (1988).The collapse of complex societies.  
New studies in archaeology. Cambridge: CambridgeUniversity Press

## Model

The model is an agent-based network model written in Python. The main files
are included under scripts/model.

## Analyses

Several analyses can be performed with the model code.

+ stochastic simulations of the network with various paramter settings.
    For details on parameters see the inline documentation in model/main.py

+ analytic calculations with equations derived in publication

## Instructions

### Installation

1. install python:

+ download https://www.python.org/downloads/
+ install
+ tick box "add to path" so python is integrated into system

2. it is highly recommended (but not required) to install a virtual environment 
insite the package root. If you do not wish to install a virtual environment,
you can skip step (2) without any consequences.

+ see: https://docs.python.org/3/library/venv.html
+ navigate with shell into project root and execute ```python -m venv env```
  this will create a virtual environment in the directory <env>
+ activate (command varies depending on OS)
  + Windows: env\Scripts\activate
  + Linux/Mac: env\bin\activate

3. install required packages: ```pip install -r docs\requirements.txt```

### Usage

to execute scripts, navigate into package root, activate environment (if using it)
and call scripts from any terminal, console or IDE.

The files __macroscopic_approximation.py__ and __stochastic_model.py__ are
very basic scripts which execute the main modules of the code.

In order to modify the output plot of __stochastic_plot.py__,
__stochastic_model.py__ needs to be executed with desired paramter settings.
The corresponding directories will appear under /data/YYYYMMDD_HHMM.
Where the time corresponds to the current system time.
These names need to be inserted into __stochastic_plot.py__. As a help,
three example runs have been already included in a data directory.

if you wish to change paramters or executed code, this needs to be done inside
the code.