from itertools import product
import sys
import Sim_dend_weigth
import numpy as np

list = (sys.argv[1]).split(" ")

# Set parameters
Case = list[0]
ww = np.round(np.float32(list[1]),5)

# Run the simulation and save results
Sim_dend_weigth.run(ww, Case)

