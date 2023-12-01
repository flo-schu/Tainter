OUTPUT=$1
PROJ_DIR="/home/$USER/projects/Tainter"
mkdir -p $OUTPUT

# set up parameters
python "$PROJ_DIR/scripts/parameter_analysis/parameters_cluster.py" $OUTPUT

# store number of created parameter files in variable
BATCHES=$(ls $OUTPUT/param* | wc -l)
echo "created $BATCHES parameter files."

# run scripts on cluster
JID=$(sbatch --parsable -a 1-$BATCHES "$PROJ_DIR/scripts/parameter_analysis/array_submit.sh" $OUTPUT)
echo "submitted parameter scan job $JID to cluster"

# combine result files
JID2=$(sbatch --dependency=afterok:$JID "$PROJ_DIR/scripts/parameter_analysis/postprocessing.sh" $OUTPUT)
echo "submitted postprocessing job $JID2"