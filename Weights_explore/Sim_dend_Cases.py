import numpy as np
import pandas as pd
import os
import sys
from neuron import h
from neuron.units import mV, ms, sec, um, us

from Edu_functions import *


Case = (sys.argv[1]).split(" ")[0]


##################################################
# Initialize Neuron model

ddir = "L5bPCmodelsEH/"

h.load_file("import3d.hoc")
h.load_file("stdgui.hoc")
h.load_file(ddir+"models/L5PCbiophys3.hoc")
h.load_file(ddir+"models/L5PCtemplate.hoc")

morphology_file = ddir+"morphologies/cell1.asc"

cell = h.L5PCtemplate(morphology_file)

####################################################
# Sections labels

Sections = [i for i in h.allsec() if ("axon" not in str(i)) and ("soma" not in str(i))]
Soma = [i for i in h.allsec() if "soma" in str(i)]

###################################################
# Load the morphology details
List = pd.read_csv("Morpho_data_Pyr_p1_.csv")
CellName = List["Cell_name"][0]

####################################################
# Initialize

# Simulation max time and integration time step
tmax = 200000 # ms
dt = 0.0125
h.dt = dt

# Number of background synapses
numNoiseInputs = 5000

# Time binning for saving
timebin = 1

# Background parameters
delay = 0

We = {"AMPA" : 0.245, "NMDA" : 0.195, "Both" : 0.08}

ww = We[Case]*us

weightE = np.random.lognormal(np.log(ww),0.25,np.sum(ee>=0)) #We[Case] #* us #microsiemens
weightI = np.random.lognormal(np.log(0.001),0.75,np.sum(ee<0)) #We[Case] #* us #microsiemens

# Different time constants for E and I synapses
# AMPA: 0.2 ms rise, 1.7 ms decay
# NMDA: 2 ms rise, 20 ms decay
# GABA: 1 ms rise, 5 ms decay
tau_AMPA = [0.2, 1.7]  # AMPA
tau_NMDA = [2, 20]     # NMDA
tau_GABA = [1, 5]      # GABA

# Spike threshold
thres = -30  # mV

####################################################
# CREATING BACKGROUND NOISE SYNAPSES UNIFORMLY DISTRIBUTED IN DENDITRIC LENGTH DISTANCE
ee, loc, dendIdx = Background(h, Sections, numNoiseInputs)

numNoiseInputs = len(ee)

# Index of excitatory synapses (important to NMDA receptors)
E_indx = np.where(ee == 0)[0]

I_indx = np.where(ee < 0)[0]

####################################################
# seeds for random inputs (background and sources)
np.random.seed(123456)

# background stimulus (NetStim) parameters
isi = 300*ms         # mean interspike time interval
num = 10000           # average number of spikes
start = 1*ms        # stimulus start time
stop = tmax*ms      # simulation stop time
noise = 1         # noise parameter (must be a value from 0 to 1)

seeds=np.random.randint(10000, size=numNoiseInputs)

#Monitoring Sections membrane potential

MonitPoints = [i for i in range(len(Sections)) if "soma" not in str(Sections[i])]

NumSen = len(MonitPoints)

####################################################
# Create synapses

# Create inputs
BackGroundstim = [createStim(h, isi=isi, num=num, start=start, noise=noise, seed=seeds[i]) for i in range(numNoiseInputs)]

# Create synapses I
synapses_GABA = [createSynapse(h, Sections, dendIdx[i], loc=loc[i], tau=tau_GABA, e=ee[i]) for i in I_indx]

# Connect inputs to synapses I
connections_GABA = [connectStim(h, synapses_GABA[i], BackGroundstim[I_indx[i]], delay=delay, weight=weightI[i]) for i in range(len(I_indx))]

if Case=="Both" or Case=="AMPA":

	synapses_AMPA = [createSynapse(h, Sections, dendIdx[i], loc=loc[i], tau=tau_AMPA, e=ee[i]) for i in E_indx]

	connections_AMPA = [connectStim(h, synapses_AMPA[i], BackGroundstim[E_indx[i]], delay=delay, weight=weightE[i]) for i in range(len(E_indx))]

if Case=="Both" or Case=="NMDA":
    
	synapses_NMDA = [createNMDA_Synapse(h, Sections, dendIdx[i], loc=loc[i], tau=tau_NMDA, e=ee[i]) for i in E_indx]

	connections_NMDA = [connectStim(h, synapses_NMDA[i], BackGroundstim[E_indx[i]], delay=delay, weight=weightE[i]) for i in range(len(E_indx))]

####################################################
# Prepare output variable
recordings = {'soma': h.Vector(),
          'sensors': [h.Vector() for i in range(NumSen)],
          'time': h.Vector()}

# Set up recordings
recordings['soma'].record(Soma[0](0.5)._ref_v)  # soma membrane potential
recordings['time'].record(h._ref_t)  # time steps

for i, sen0 in enumerate(MonitPoints):
	recordings['sensors'][i].record(Sections[sen0](0.5)._ref_v)

####################################################
# Data saving directory and filename
savdir = "Sim_MonitMembranePot_"+Case+"/"

isExist = os.path.exists(savdir)

if not isExist:
	# Create a new directory because it does not exist
	os.makedirs(savdir)

filename = savdir+"/Sim_"+Case+"_w_"+str(We[Case])+"_tmax_"+str(tmax)+"_Cell_"+CellName+"_timebin_"+str(timebin)+"_numinputs_"+str(numNoiseInputs)+"_ISInoise_"+str(isi)+"_avgnumspikes_"+str(num)+".csv"

####################################################
# Run simulation

if os.path.isfile(filename) == False:

	h.finitialize(-65*mV)

	h.continuerun(tmax*ms)

	for k, v in recordings.items():
		if k == 'soma' or k == 'time':
			recordings[k] = np.array(list(recordings[k]))
		else:
			recordings[k] = [np.array(list(recordings[k][i])) for i in range(len(recordings[k]))]

	TimeSeries = np.zeros([len(recordings["time"]),NumSen+1],dtype=np.int8)

	TimeSeries[:,0] = 1*(recordings["soma"]>thres)

	for i in range(0,NumSen,1):

		TimeSeries[:,i+1] = 1*(recordings['sensors'][i]>thres)

	Tw = int(timebin/dt)
	TT = len(recordings['time'])
	NT = int(TT/Tw)

	TSer = np.zeros([NT,NumSen+1],dtype=np.int8)

	for i in range(NT-1):

		TSer[i,:] = 1*(np.sum(TimeSeries[Tw*i:Tw*(i+1),:],axis=0)>0)

	Labels = [str(Soma[0])]
	Labels += [str(i) for i in Sections]

	Data = pd.DataFrame(columns = Labels)

	for i in range(len(Labels)):

		Data[Data.columns[i]] = TSer[:,i]
	    
	Data.to_csv(filename, index=False)

