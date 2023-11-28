#!/usr/bin/env bash
#SBATCH --job-name=tainter-postprocessing_parscan                      # name of job
#SBATCH --time=0-1:00:00                                       # maximum time until job is cancelled
#SBATCH --cpus-per-task=1                                      # number of nodes requested
#SBATCH --mem-per-cpu=32G                                      # memory per cpu requested
#SBATCH --mail-type=begin                                      # send mail when job begins
#SBATCH --mail-type=end                                        # send mail when job ends
#SBATCH --mail-type=fail                                       # send mail if job fails
#SBATCH --mail-user=florian.schunck@uos.de                     # email of user
#SBATCH --output=/home/staff/f/fschunck/logs/job-%x-%j.out  # output file of stdout messages
#SBATCH --error=/home/staff/f/fschunck/logs/job-%x-%j.err   # output file of stderr messages


OUTPUT=$1
PROJ_DIR="/home/staff/f/$USER/projects/tainter"

# prepare environment, e.g. set path
module purge

# activate conda environment
# module load Anaconda3
# source activate tainter
# module unload Anaconda3

echo "postprocessing results of parameter scan ..."


python "$PROJ_DIR/scripts/parameter_analysis/process_cluster.py" $OUTPUT
python "$PROJ_DIR/scripts/parameter_analysis/plot_cluster.py" $OUTPUT


