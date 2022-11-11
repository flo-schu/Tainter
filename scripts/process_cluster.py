import sys
import os
import numpy as np
from tainter.cluster.parameter_scan_odeint import process_output
from tainter.f4_plot_parameter_analysis import fig4_parameter_analysis

output = sys.argv[1]
data = process_output(directory=output)

data_file = os.path.join(output, "results_combined_cluster.txt")
np.savetxt(data_file, data)


fig4_parameter_analysis(
    data_file=data_file,
    multiline_steps=[]
)