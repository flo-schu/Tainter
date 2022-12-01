# run parameter analysis on cluster

the easiest way to run a parameter analysis is to adjust the parameter 
ranges in `scripts/parameters_cluster.py` and then execute
`source scripts/parameter_analysis/parameter_analysis.sh output_directory`
where <output_directory> specifies the path where all output is stored

this executes all steps below sequentially. Which can alternatively be executed
individually.

## step 1: set up parameters for cluster 

modify script `scripts/parameter_analysis/parameters_cluster.py`, particularly the ranges of parameters, step size and batch size

then, run the script `python scripts/parameter_analysis/parameter_cluster output_directory` where <output_directory> specifies the path where the parameters should be stored. This script creates a parameter file for each batch to be run on the cluster

## step 2: run the parameter scan

execute for SLURM scheduling system `sbatch -a 1-N scripts/parameter_analysis/array_submit.sh output_directory`, where <output_directory> is the same as specified above and <N> is the number of batches (the number of files prepared in step 1)

## step 3: process files

to speed up plotting, all result files from step 2 are combined in a single text file. This is done with a call to `python scripts/parameter_analysis/process_cluster.py output_directory`, where <output_directory> is again the same folder, this will take a while, even after the progress bar is complete, depending on the amount of files

## step 4: plot analysis

to plot the results execute `python scripts/parameter_analysis/plot_cluster.py output_directory`. 
Again <output_directory> is the same folder. This will create a plot under
`scripts/` named `fig4_parameter_analysis.png`


