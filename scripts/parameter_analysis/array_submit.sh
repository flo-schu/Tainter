#!/usr/bin/env bash
#SBATCH --job-name=tainter-parameter_scan                      # name of job
#SBATCH --time=0-00:15:00                                      # maximum time until job is cancelled
#SBATCH --ntasks=1                                             # number of tasks
#SBATCH --cpus-per-task=1                                      # number of nodes requested
#SBATCH --mem-per-cpu=4G                                       # memory per cpu requested
#SBATCH --mail-type=begin                                      # send mail when job begins
#SBATCH --mail-type=end                                        # send mail when job ends
#SBATCH --mail-type=fail                                       # send mail if job fails
#SBATCH --mail-user=florian.schunck@uos.de                     # email of user
#SBATCH --output=/home/staff/f/fschunck/logs/job-%x-%A-%a.out  # output file of stdout messages
#SBATCH --error=/home/staff/f/fschunck/logs/job-%x-%A-%a.err   # output file of stderr messages


OUTPUT=$1
PROJ_DIR="/home/staff/f/$USER/projects/tainter"

mkdir -p $OUTPUT

# prepare environment, e.g. set path

# activate conda environment
# source activate tainter


echo "processing chunk $SLURM_ARRAY_TASK_ID ..."


python -W ignore "$PROJ_DIR/tainter/cluster/parameter_scan_odeint.py" \
    $OUTPUT \
    $SLURM_ARRAY_TASK_ID \
    p_e rho c

