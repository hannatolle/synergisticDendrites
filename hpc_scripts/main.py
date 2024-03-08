from itertools import product
import sys
import simulations_dendritic_dyn


# Define indices of input synapses (indices of 2 out of 193 dendrite sections)
params = list(product(range(193), range(193)))

# Fetch job_id from command-line argument and set parameters
path2home = sys.argv[1]
job_id = int(sys.argv[2])
nn = params[job_id-1]

# Run the simulation and save results
simulations_dendritic_dyn.run(nn, path2home)
