#!/bin/bash

#$ -S /bin/bash
#$ -wd /home/schunckf
#$ -N tainter
#$ -l h_rt=00:30:00
#$ -l h_vmem=1G
#$ -o /work/$USER/$JOB_NAME/$JOB_ID/log_$TASK_ID.txt
#$ -j y
#$ -binding linear:1
output_dir="/work/$USER/$JOB_NAME/$JOB_ID"
mkdir -p "$output_dir"

module load Anaconda3/5.3.0
source activate .conda/envs/tainter

echo "processing chunk $SGE_TASK_ID ..."

python -W ignore ./tainter/parameter_analysis_20200713/parameter_scan_odeint.py "./tainter/parameter_analysis_20200713/params/chunk_$SGE_TASK_ID.txt" "$output_dir" "$SGE_TASK_ID"

echo "chunk successfully executed"
source deactivate
