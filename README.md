# Tainter

Network model of simple societies inspired by Joseph Tainter's theory of the
collapse of complex societies.

Tainter, J. A. (1988).The collapse of complex societies.  
New studies in archaeology. Cambridge: CambridgeUniversity Press

## installation

Obtain the code and change into directory

```bash
git clone git@github.com:flo-schu/tainter
cd tainter
```

Create environment, activate it and install model package.
the [datalad] option installs the datalad packages which are necessary to 
download the dataset.

```bash
conda create -n tainter
conda activate tainter
conda install python=3.9
pip install -e .[datalad] 
```

Install the pre-simulated dataset from the open-science foundation (osf.io) 
into the folder_publication data

```bash
datalad clone https://osf.io/u897c/ publication_data
```

You should be done. All anylses from the paper can be conducted by executing
the jupyter notebook `scripts/analysis.ipynb`

```bash
jupyter lab scripts/analysis.ipynb
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