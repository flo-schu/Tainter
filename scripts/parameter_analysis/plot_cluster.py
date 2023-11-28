import sys
import os
from tainter.f4_plot_parameter_analysis import fig4_parameter_analysis

output = sys.argv[1]
data_file = os.path.join(output, "results_combined_cluster.txt")
multiline_steps = [1.0, 1.2, 1.35, 1.5, 1.65, 1.85, 2.1, 2.2, 2.3]
fig = fig4_parameter_analysis(data_file, multiline_steps=multiline_steps)
fig.savefig(os.path.join(output, "fig4_parameter_analysis.png"))
fig.savefig("scripts/fig4_parameter_analysis.png")