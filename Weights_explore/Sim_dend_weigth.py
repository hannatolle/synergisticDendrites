import numpy as np
import pandas as pd
import os
from neuron import h
from neuron.units import mV, ms, sec, um, us

from Edu_functions import *


def run(ww,Case="Both"):
    """
    Simulates pyramidal neuron activity upon
    5000 background synapses.

    Parameters
    ----------
    ww: Uniform weigth value for synapses
    
    Case: Define receptors configuration.
    - "both" for both excitatory receptors AMPA and NMDA, 
    - "AMPA" for using only AMPA receptors,
    - "NMDA" for using only NMDA receptors
    
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
    tmax = 10000  # ms
    dt = 0.0125
    h.dt = dt

    # Number of background synapses
    numNoiseInputs = 5000

    # Time binning for saving
    timebin = 1

    # Background parameters
    delay = 0
    
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
    # CREATING BACKGROUND NOISE SYNAPSES UNIFORMLY DISTRIBUTED IN DENDITRIC LENGTH DISTANCE
    ee, loc, dendIdx = Background(h, Sections, numNoiseInputs, pp=pps)

    numNoiseInputs = len(ee)
    
    #weight = np.random.lognormal(np.log(ww),500*ww,numNoiseInputs) #We[Case] #* us #microsiemens
    weightE = np.random.lognormal(np.log(ww),0.55,np.sum(ee>=0)) #We[Case] #* us #microsiemens
    weightI = np.random.lognormal(np.log(0.001),0.5,np.sum(ee<0)) #We[Case] #* us #microsiemens

    # Index of excitatory synapses (important to NMDA receptors)
    E_indx = np.where(ee == 0)[0]

    I_indx = np.where(ee < 0)[0]

    ####################################################
    # seeds for random inputs (background and sources)
    np.random.seed(123456)

    # background stimulus (NetStim) parameters
    isi = 400*ms         # mean interspike time interval
    num = 10000           # average number of spikes
    start = 1*ms        # stimulus start time
    stop = tmax*ms      # simulation stop time
    noise = 1         # noise parameter (must be a value from 0 to 1)

    seeds=np.random.randint(10000, size=numNoiseInputs)

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
                  'time': h.Vector()}

    # Set up recordings
    recordings['soma'].record(Soma[0](0.5)._ref_v)  # soma membrane potential
    recordings['time'].record(h._ref_t)  # time steps

    ####################################################
    # Data saving directory and filename
    savdir = "Activity_"+Case+"/"

    isExist = os.path.exists(savdir)

    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(savdir)

    filename = savdir+"/Sim_"+Case+"_w_"+str(ww)+"tmax_"+str(tmax)+"_Cell_"+CellName+"_timebin_"+str(timebin)+"_numinputs_"+str(numNoiseInputs)+"_ISInoise_"+str(isi)+"_avgnumspikes_"+str(num)+".csv"

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

        TimeSeries = np.zeros([len(recordings["time"])], dtype=np.int8)

        TimeSeries[1:] = (np.sign(np.diff((1*(recordings["soma"] > thres)))) > 0)

        Tw = int(timebin/dt)
        TT = len(recordings['time'])
        NT = int(TT/Tw)

        TSer = np.zeros([1,NT], dtype=np.int16)

        for i in range(NT-1):
            
            TSer[0,i] = 1*(np.sum(TimeSeries[Tw*i:Tw*(i+1)]))

        Labels = [str(Soma[0])]

        Data = pd.DataFrame(columns=Labels)

        for i in range(len(Labels)):
            
            Data[Data.columns[i]] = TSer[i,:]
            
        Data.to_csv(filename, index=False)


if __name__ == '__main__':
    raw = run(ww=0.0003, Case='both')
