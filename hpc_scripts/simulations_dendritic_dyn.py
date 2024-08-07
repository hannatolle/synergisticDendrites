import numpy as np
import pandas as pd
import os
from neuron import h
from neuron.units import mV, ms, sec, um, us

from Edu_functions import *


def run(nn, path2home):
	"""
	Simulates pyramidal neuron activity upon stimulation via 2 strong dendritic
	synapses and 5000 weak background synapses.

	Parameters
	----------
	nn: tuple
	Indices of the 2 dendritic sections where the strong synapses are
	located.

	path2home: string
	Filepath to the HOME directory.

	Returns
	-------
	raw: mne.io.Raw
	MNE Raw object with results of the simulation in a 10-20 EEG montage.
	"""
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
	tmax = 100  # ms
	dt = 0.0125
	h.dt = dt

	# Number of background synapses
	numNoiseInputs = 5000

	# Time binning for saving
	timebin = 1

	# Background parameters
	delay = 0
	weight = 0.05*us

	# Different time constants for E and I synapses
	# AMPA: 0.2 ms rise, 1.7 ms decay
	# NMDA: 2 ms rise, 20 ms decay
	# GABA: 1 ms rise, 5 ms decay
	tau_AMPA = [0.2, 1.7]  # AMPA
	tau_NMDA = [2, 20]     # NMDA
	tau_GABA = [1, 5]      # GABA

	# Spike threshold
	thres = -30  # mV

	#Excitatory and inhibitory proportions
	pps = [0.8,0.2]

	####################################################
	# CREATING BACKGROUND NOISE SYNAPSES UNIFORMLY DISTRIBUTED IN DENDITRIC SURFACE AREA
	ee, loc, dendIdx = Background(h, Sections, numNoiseInputs, pp=pps)

	#Background parameters
	delay = 0

	numNoiseInputs = len(ee)

	#Excitatory average weight depending on neuroreceptors
	We = {"AMPA" : 0.000592, "NMDA":0.00053, "Both":0.000287}

	#Define the case
	Case = "Both"

	ww = We[Case]

	#Standard deviation for the log-Normal distribution 
	weStd = 0.55 #Excitatory
	wiStd = 0.5 #Inhibitory

	weightE = np.random.lognormal(np.log(ww),weStd,np.sum(ee>=0)) #We[Case] #* us #microsiemens
	weightI = np.random.lognormal(np.log(0.001),wiStd,np.sum(ee<0)) #We[Case] #* us #microsiemens

	# Index of excitatory synapses (important to NMDA receptors)
	E_indx = np.where(ee == 0)[0]

	I_indx = np.where(ee < 0)[0]

	###############################################################
	# Selecting sources inputs positions from a List of candidates 
	Data = pd.read_csv("Inputs_info_"+Case+".csv")

	dendIndxInp = Data.InpIndx.values
	dendlocInp = Data['loc'].values

	#Random choice
	IndB = np.random.choice(Data[Data.Category == 'Basal'].index,2,replace=False)
	IndM = np.random.choice(Data[Data.Category == 'Mid'].index,2,replace=False)
	IndA = np.random.choice(Data[Data.Category == 'Api'].index,2,replace=False)

	Inpidx_sel = np.array([dendIndxInp[IndB],dendIndxInp[IndM],dendIndxInp[IndA]]).flatten()
	Inploc_sel = np.array([dendlocInp[IndB],dendlocInp[IndM],dendlocInp[IndA]]).flatten()

	#Input weight
	WInp = 0.004

	numSour = len(Inpidx_sel)

	########################################################
	# seeds for random inputs (background and sources)
	np.random.seed(123456)

	# background stimulus (NetStim) parameters
	isi = 400*ms         # mean interspike time interval
	num = 50+tmax*ms/isi           # average number of spikes
	start = 1*ms        # stimulus start time
	stop = tmax*ms      # simulation stop time
	noise = 1         # noise parameter (must be a value from 0 to 1)

	seeds = np.random.randint(10000, size=numNoiseInputs)

	# input stimulus (NetStim) parameters
	isiInp = 100*ms         # mean interspike time interval
	StiDur = tmax #ms #Stimulus average duration
	numInp = (StiDur/isiInp)  + 50          # average number of spikes
	startInp = 5000*ms        # stimulus start time
	stop = tmax*ms      # simulation stop time

	seedsInp=np.random.randint(10000, size=numSour)

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

	# create inputs
	InputsStim = [createStim(h,isi=isiInp, num=numInp, start=startInp, noise=noise, seed=seedsInp[i]) for i in range(numSour)]

	# create synapses I
	Inpsynapses_AMPA = [createSynapse(h,Sections,Inpidx_sel[i], loc=Inploc_sel[i], tau=tau_AMPA, e=0) for i in range(numSour)]
	Inpsynapses_NMDA = [createNMDA_Synapse(h,Sections,Inpidx_sel[i], loc=Inploc_sel[i], tau=tau_NMDA, e=0) for i in range(numSour)]

	# connect inputs to synapses E 
	Inpconnections_AMPA = [connectStim(h,Inpsynapses_AMPA[i], InputsStim[i], delay=delay, weight=WInp) for i in range(numSour)]
	Inpconnections_NMDA = [connectStim(h,Inpsynapses_NMDA[i], InputsStim[i], delay=delay, weight=WInp) for i in range(numSour)]

	####################################################
	# Prepare output variable
	recordings = {'soma': h.Vector(),
		      'input': [h.Vector() for i in range(numSour)],
		      'inputTime': [h.Vector() for i in range(numSour)],
		      'time': h.Vector()}

    #Record every 10*dt
    ddt = 10*dt

    # set up recordings
    recordings['Vsoma'].record(Soma[0](0.5)._ref_v,ddt) # soma membrane potential
    
    recordings['time'].record(h._ref_t,ddt) # time steps

	# set up recordings
	recordings['soma'].record(Soma[0](0.5)._ref_v) # soma membrane potential
	recordings['time'].record(h._ref_t) # time steps

	#Recording sources inputs 
	for i, dend0 in enumerate(range(numSour)):
		Inpconnections_AMPA[dend0].record(recordings['inputTime'][i], recordings['input'][i])

	####################################################
	# Data saving directory and filename
	savdir = path2home+"/SavedData/"

	isExist = os.path.exists(savdir)

	if not isExist:
		# Create a new directory because it does not exist
		os.makedirs(savdir)

	fileInp = "Inputs_selected_Idx"+str(nn[0])+".txt"

	np.savetxt(fileInp, np.vstack([Inpidx_sel,Inploc_sel]).T, delimiter=',')

	filename = savdir+"/Sim_6inputs_"+str(nn[0])+"_tmax_"+str(tmax)+"_Cell_"+CellName+"_timebin_"+str(timebin)+"_numinputs_"+str(numNoiseInputs)+"_ISInoise_"+str(isi)+"_avgnumspikes_"+str(num)+".csv"

	####################################################
	# Run simulation

	if os.path.isfile(filename) == False:

		print("Running")

		h.finitialize(-65*mV)

		h.continuerun(tmax*ms)

		for k, v in recordings.items():
			if k == 'soma' or k == 'time':
				recordings[k] = np.array(list(recordings[k]))
			else:
				recordings[k] = [np.array(list(recordings[k][i])) for i in range(len(recordings[k]))]

		InputsTS = np.zeros([numSour, tmax], dtype=bool)

		for i in range(numSour):
		    
			try:
				inptim = np.int32(np.floor(recordings["inputTime"][i]))
			except:
				inptim = []
			    
			if len(inptim) > 0:
				InputsTS[i, inptim] = 1

		TimeSeries = np.zeros([len(recordings["time"]), numSour+1], dtype=np.int8)

		TimeSeries[1:, 0] = (np.sign(np.diff((1*(recordings["soma"] > thres)))) > 0)

		Tw = int(timebin/(ddt))
        TT = len(recordings['time'])
        NT = int(TT/Tw)

		TSer = np.zeros([NT, numSour+1], dtype=np.int16)

		for i in range(NT-1):
		    
			TSer[i, :] = 1*(np.sum(TimeSeries[Tw*i:Tw*(i+1), :], axis=0))

		TSer[:, 1:] = InputsTS.T

		Labels = [str(Soma[0])]
		Labels += [str(Sections[j]) for j in IndxSour]

		Data = pd.DataFrame(columns=Labels)

		for i in range(len(Labels)):
		    
			Data[Data.columns[i]] = TSer[:, i]

		for i in range(len(Labels)):

			Data[Data.columns[i]] = TSer[:, i]
		    
		Data.to_csv(filename, index=False)


if __name__ == '__main__':
	raw = run(nn=[1, 2], path2home='.')
