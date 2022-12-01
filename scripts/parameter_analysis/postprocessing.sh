#!/usr/bin/env bash
#SBATCH --job-name=postprocessing_parscan              # name of job
#SBATCH --time=0-1:00:00                               # maximum time until job is cancelled
#SBATCH --cpus-per-task=1                              # number of nodes requested
#SBATCH --mem-per-cpu=4G                               # memory per cpu requested
#SBATCH --output=/work/%u/tainter/logs/%x-%A-%a.out    # output file of stdout messages
#SBATCH --error=/work/%u/tainter/logs/%x-%A-%a.err     # output file of stderr messages
#SBATCH --mail-type=begin                              # send mail when job begins
#SBATCH --mail-type=end                                # send mail when job ends
#SBATCH --mail-type=fail                               # send mail if job fails
#SBATCH --mail-user=florian.schunck@ufz.de             # email of user


OUTPUT=$1
PROJ_DIR="/home/$USER/projects/Tainter"

# prepare environment, e.g. set path
module purge

# activate conda environment
module load Anaconda3
source activate tainter
module unload Anaconda3

echo "postprocessing results of parameter scan ..."


python "$PROJ_DIR/scripts/parameter_analysis/process_cluster.py" $OUTPUT
python "$PROJ_DIR/scripts/parameter_analysis/plot_cluster.py" $OUTPUT


