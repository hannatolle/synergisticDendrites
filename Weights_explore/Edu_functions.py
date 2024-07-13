import numpy as np
import pandas as pd
import os
from neuron import h
from neuron.units import mV, ms, sec, um, us

def Background(h,Sections,numNoiseInputs,pp=[0.8,0.2],Save=False,Plot=False):
    """
    CREATING BACKGROUND NOISE SYNAPSES UNIFORMLY DISTRIBUTED BY DENDITRIC LENGTH
    
    Params:
    - Sections: List of section labels
    - numNoiseInputs: Number of background noise inputs
    """

    Asections = [np.pi*eval("h."+str(i)).psection()["morphology"]["L"]*np.mean(eval("h."+str(i)).psection()["morphology"]["diam"]) for i in Sections]
    TotalArea = np.sum(Asections)
    Synapdensity = numNoiseInputs/TotalArea

    #Lsections = [eval("h."+str(i)).psection()["morphology"]["L"] for i in Sections]
    #TotalLength = np.sum(Lsections)
    #Synapdensity = numNoiseInputs/TotalLength
    

    NumSynap_perSec = np.int32(np.round(Synapdensity*np.array(Asections)))

    dendIdx = np.repeat(np.arange(0,len(Sections)),NumSynap_perSec)

    numNoiseInputs = len(dendIdx)
    
    ee,loc = EI_Synaptic_info(numNoiseInputs,pp=pp)
    
    if Save==True:
        """Save background inputs info"""

        from scipy.io import savemat

        m = {"Indx":dendIdx,"loc":loc,"EI":ee}

        savemat("DATA/"+"RandomInputs_"+"simulation_"+str(tmax)+"_Cell_"+CellName+"_timebin_"+str(timebin)+"_numinputs_"+str(numNoiseInputs)+".mat", m)

    if Plot==True:
        """Plot background input distribution"""
        
        Plot3D("Pyramidal_1",inputsIdx=dendIdx,loc=loc)

    return ee,loc,dendIdx

def createSynapse(h,dend,dendIdx, loc=0.5, tau=[2*ms, 4*ms], e=0*mV):
    """Creates an excitatory synapse on the dendrite given by Idx."""
    syn = h.Exp2Syn(dend[dendIdx](loc))
    syn.tau1 = tau[0]
    syn.tau2 = tau[1]
    syn.e = e
    return syn 

def createNMDA_Synapse(h,dend,dendIdx, loc=0.5, tau=[2*ms, 20*ms], e=0*mV):
    """Creates an excitatory synapse on the dendrite given by Idx."""
    syn = h.Exp2SynNMDA(dend[dendIdx](loc))
    syn.tau1 = tau[0]
    syn.tau2 = tau[1]
    syn.e = e
    return syn 

def createStim(h,isi=20*ms, num=100, start=1*ms, noise=1, seed=9):
    stim = h.NetStim()
    stim.interval = isi
    stim.number = num
    stim.start = start
    stim.noise = noise
    stim.noiseFromRandom123(seed, 0, 0)
    stim.seed(seed)
    
    return stim

def connectStim(h,syn, stim, delay=1*ms, weight=0.4):
    conn = h.NetCon(stim, syn)
    conn.delay = delay
    conn.weight[0] = weight
    
    return conn

def createStim(h,isi=20*ms, num=100, start=1*ms, noise=1, seed=9):
        stim = h.NetStim()
        stim.interval = isi
        stim.number = num
        stim.start = start
        stim.noise = noise
        stim.noiseFromRandom123(seed, 0, 0)
        stim.seed(seed)
        return stim

def EI_Synaptic_info(numNoiseInputs,pp = [0.8,0.2]):
    
    es = [0,-80]

    ee = np.random.choice(es,numNoiseInputs,p=pp)

    loc = np.round(np.random.uniform(0,1,numNoiseInputs),3)
    
    return ee,loc

def Select_Noise_sections(List,Sections,Cell="Pyr_p1",typ=0,rangeL=[0,1500],numNoiseInputs=100):
    
    if typ==0:
        Cell0 = List[(List['Cell_name']==Cell)&((List['Labels'].str.contains("dend"))|(List['Labels'].str.contains("apic")))]
    else:
         Cell0 = List[(List['Cell_name']==Cell)&(List['Labels'].str.contains(typ))]
       
    #For background noise
    rangeL = [0,1500]

    SecNames = Cell0[(Cell0['PathLength']>rangeL[0])&(Cell0['PathLength']<rangeL[1])].Labels.values

    dendIdx0 = []

    for i in range(len(Sections)):
        sec = Sections[i]
        if str(sec) in SecNames:
            dendIdx0.append(i)
    
    dendIdx = np.random.choice(dendIdx0,numNoiseInputs)
    
    return dendIdx

