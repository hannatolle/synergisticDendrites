import numpy as np
import pandas as pd
import os
from neuron import h
from neuron.units import mV, ms, sec, um, us

from For_Hanna/Edu_functions import *


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
    tmax = 200000  # ms
    dt = 0.0125
    h.dt = dt

    # Number of input synapses (sources)
    numSour = 2

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

    ####################################################
    # CREATING BACKGROUND NOISE SYNAPSES UNIFORMLY DISTRIBUTED IN DENDITRIC LENGTH DISTANCE
    ee, loc, dendIdx = Background(h, Sections, numNoiseInputs)

    numNoiseInputs = len(ee)

    # Index of excitatory synapses (important to NMDA receptors)
    E_indx = np.where(ee == 0)[0]

    I_indx = np.where(ee < 0)[0]

    ####################################################
    # Selecting input synapse positions

    IndxSour = np.array([nn[0], nn[1]])

    ####################################################
    # seeds for random inputs (background and sources)
    np.random.seed(123456)

    # background stimulus (NetStim) parameters
    isi = 100*ms         # mean interspike time interval
    num = 10000           # average number of spikes
    start = 1*ms        # stimulus start time
    stop = tmax*ms      # simulation stop time
    noise = 1         # noise parameter (must be a value from 0 to 1)

    seeds=np.random.randint(10000, size=numNoiseInputs)

    # input stimulus (NetStim) parameters
    isiInp = 100*ms         # mean interspike time interval
    numInp = 10000           # average number of spikes
    startInp = 1*ms        # stimulus start time
    stop = tmax*ms      # simulation stop time

    seedsInp = np.random.randint(10000, size=len(IndxSour))

    ####################################################
    # Create synapses

    # Create synapses I
    synapses_GABA = [createSynapse(h, Sections, dendIdx[i], loc=loc[i], tau=tau_GABA, e=ee[i]) for i in I_indx]
    synapses_AMPA = [createSynapse(h, Sections, dendIdx[i], loc=loc[i], tau=tau_AMPA, e=ee[i]) for i in E_indx]
    synapses_NMDA = [createNMDA_Synapse(h, Sections, dendIdx[i], loc=loc[i], tau=tau_NMDA, e=ee[i]) for i in E_indx]

    # Create inputs
    BackGroundstim = [createStim(h, isi=isi, num=num, start=start, noise=noise, seed=seeds[i]) for i in range(numNoiseInputs)]

    # Connect inputs to synapses I
    connections_GABA = [connectStim(h, synapses_GABA[i], BackGroundstim[I_indx[i]], delay=delay, weight=weight) for i in range(len(I_indx))]
    connections_AMPA = [connectStim(h, synapses_AMPA[i], BackGroundstim[E_indx[i]], delay=delay, weight=weight) for i in range(len(E_indx))]
    connections_NMDA = [connectStim(h, synapses_NMDA[i], BackGroundstim[E_indx[i]], delay=delay, weight=weight) for i in range(len(E_indx))]

    # Create inputs
    InputsStim = [createStim(h, isi=isiInp, num=numInp, start=startInp, noise=noise, seed=seedsInp[i]) for i in range(len(IndxSour))]

    # Create synapses I
    Inpsynapses_AMPA = [createSynapse(h, Sections, IndxSour[i], loc=1, tau=tau_AMPA, e=0) for i in range(len(IndxSour))]
    Inpsynapses_NMDA = [createNMDA_Synapse(h, Sections, IndxSour[i], loc=1, tau=tau_NMDA, e=0) for i in range(len(IndxSour))]

    # Connect inputs to synapses E
    Inpconnections_AMPA = [connectStim(h, Inpsynapses_AMPA[i], InputsStim[i], delay=delay, weight=150*us) for i in range(len(IndxSour))]
    Inpconnections_NMDA = [connectStim(h, Inpsynapses_NMDA[i], InputsStim[i], delay=delay, weight=150*us) for i in range(len(IndxSour))]

    ####################################################
    # Prepare output variable
    recordings = {'soma': h.Vector(),
                  'input': [h.Vector() for i in range(numSour)],
                  'inputTime': [h.Vector() for i in range(numSour)],
                  'time': h.Vector()}

    # Set up recordings
    recordings['soma'].record(Soma[0](0.5)._ref_v)  # soma membrane potential
    recordings['time'].record(h._ref_t)  # time steps

    # Record from input sources
    for i, dend0 in enumerate(range(len(IndxSour))):
        Inpconnections_AMPA[dend0].record(recordings['inputTime'][i], recordings['input'][i])

    ####################################################
    # Data saving directory and filename
    savdir = path2home+"/SavedData/"

    isExist = os.path.exists(savdir)

    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(savdir)

    filename = savdir+"/Sim_pairs_"+str(nn[0])+"_"+str(nn[1])+"_tmax_"+str(tmax)+"_Cell_"+CellName+"_timebin_"+str(timebin)+"_numinputs_"+str(numNoiseInputs)+"_ISInoise_"+str(isi)+"_avgnumspikes_"+str(num)+".csv"

    ####################################################
    # Run simulation

    if os.path.isfile(filename) == False:

        print("Running Section ", nn[0], "-", nn[1], " Realization ")
        
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

        Tw = int(timebin/dt)
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
