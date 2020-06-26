#!/bin/bash

#$ -S /bin/bash
#$ -wd /home/schunckf
#$ -N tainter
#$ -l h_rt=00:05:00
#$ -l h_vmem=1G
#$ -o /work/$USER/$JOB_NAME/$JOB_ID/log.txt
#$ -j y
#$ -binding linear:1
output_dir="/work/$USER/$JOB_NAME/$JOB_ID"
mkdir -p "$output_dir"

module load Anaconda3
source activate .conda/envs/tainter

echo "processing chunk $SGE_TASK_ID ..."

python ./tainter/scripts/cluster/paramter_scan_p_cluster_20200626.py ./tainter/scripts/cluster/params.csv $SGE_TASK_ID $output_dir


source deactivate
