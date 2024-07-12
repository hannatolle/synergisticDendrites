import numpy as np
from matplotlib import pyplot, cm
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
    

def BranchingLevel(List):
    """
    Give the level (number of bifurcations that occurs until reach a specific section) of each sections and branching point in the dendritic tree from Morphology information Pandas Dataframe.
    
    Param:
    - List: Dataframe with information about sections, path from soma to section and pathslenghts.
    
    Return:
        Return two arrays:
        -BifLev: Array with section were a bifurcation (branching) take place and its level in the dendritic tree hierarchy.
        -SecLev: Array indicating Indx of each section and it's level in the dendritic tree hierarchy.
    """
    
    SectionsPath = [list(np.int32(List.Paths.values[i].split("[[")[1].split("]")[0].split(','))) for i in range(1,len(List))]

    Ext = list(np.where(List.Extremes.values[1:]==True)[0])

    Paths = [SectionsPath[i] for i in Ext]

    List['PathLength'] = List.PathLength - List.PathLength[0]

    Bp = List.Num_Branchs.values

    BifLev = []

    SecLev = []

    for i in range(len(Paths)):

        Branch = list(Bp[Paths[i]])

        secl = []

        if len(Branch)>2:

            lev = []
            nn = 1

            for ni in Branch:

                if ni==2:
                    lev.append(nn)
                    nn+= 1

                elif ni==0:                
                    lev.append(lev[-1])

                else:
                    lev.append(0)

            BifLev += [(Paths[i][j],lev[j]) for j in range(len(lev)) if Branch[j]==2]

            secl = [(Paths[i][j],lev[j]) for j in range(len(lev))]

            SecLev += secl

        else:

            SecLev += [(Paths[i][1],1)]

    BifLev = np.array(list(set(BifLev)))

    BifLev = np.array(sorted(BifLev, key= lambda x: x[0]))

    SecLev = np.array(sorted(list(set(SecLev)), key = lambda x: x[0]))
    
    return BifLev,SecLev

def Inputs_dist(List,ipls=150,Li=0,Lf=0):
    """
    Identify all posible inputs locations between Li and Lf micras from the soma, equidistantly distributed at ipls micras.
    Param:
    - List: Pandas dataframe with morphological data as PathLength between a section and soma.
    - Li: Minimum pathlenght requiered. If 0, will take the minimmum pathlength available in the morphology
    - Lf: Maximum pathlenght requiered. If 0, will take the maximum pathlength available in the morphology
    - ipls: Distance between inputs. Default is 150 micras.
    
    Return: 
        Will return 3 arrays. 
            First: List of section index, 
            Second: List of input localization in that section. 
            Third: Distance between input and soma.
    """
    
    Sections = list(List.Labels.values[1:])
    SectionsPath = [list(np.int32(List.Paths.values[i].split("[[")[1].split("]")[0].split(','))) for i in range(1,len(List))]
    PathLength = List.PathLength.values[1:]

    if Li==0:
        Li = np.min(PathLength)
    
    if Li//ipls<0:
        Li = ipls
    
    MinMul = int(Li//ipls)

    if Lf==0:
        Lf = np.max(PathLength)

    MaxMul = int(Lf//ipls)

    mul = np.arange(MinMul,MaxMul+1,1)

    InpSecIndx = []
    InpSecloc = []
    InpDist = []

    # Multiple i of ipls
    for i in mul:

        # Sections at i*ipls micras or more to the soma 
        Sec0 = list(np.where((PathLength>ipls*i))[0])

        Lsections = [eval("h."+str(k)).psection()["morphology"]["L"] for k in Sections]

        # Sections which start at distance less than i*ipls and at distance greater than
        Sec1 = [Sec0[j] for j in range(len(Sec0)) if (PathLength[Sec0[j]]-Lsections[Sec0[j]])<ipls*i]

        l1 = np.array([Lsections[Sec1[k]] for k in range(len(Sec1))])
        l2 = np.array([PathLength[Sec1[k]] for k in range(len(Sec1))])

        # localization of distance i*ipls inside this section
        loc = np.round(1-(l2-ipls*i)/l1,3)
        
        InpSecIndx += Sec1
        InpSecloc += list(loc)
        InpDist += list(np.ones(len(Sec1))*ipls*i)
    
    InpDia = [np.round(np.mean(eval("h."+str(Sections[k])).psection()["morphology"]["diam"]),3) for k in InpSecIndx]
    
    return np.array(InpSecIndx),np.array(InpSecloc),np.array(InpDist),np.array(InpDia)


def Sample_inputs(List,dist_bet=100,dis_start=0,dis_end=0,num=2):
    """
    Indentify the number of possible inputs between a path length interval dis_start and dis_end that
    can be selected in the dendritic when they are equidistantly distributed. Then sample a number of 
    inputs for each distance and each branching level present between the possible inputs.
    
    The distance of inputs will be always integer multiples of dist_bet.
    Branching level refeer to number of bifurcations that exist between the soma and the given input point.
    
    Params:
    -List: Morphology info DataFrame.
    -dist_bet: Distance between inputs (Default=100 micras)
    -dis_start: Minimum distance of sampling (if 0, will take the minimum pathlenght, usually the lengh of shortes section).
    -dis_end: Maximum distance of sampling (if 0, will take the maximum pathlenght present in the tree).
    -num: Number of samples for each Distance and Branching Level.
    
    Return
        An nd-array with List of Inputs Info: 
            Column 1: Index, 
            Column 2: Input localization in the section, 
            Column 3: Input Distance from soma,
            Column 4: Input Level 
    """
    BifLev,SecLev = BranchingLevel(List)
    
    InpSecIndx,InpSecloc,InpDist,InpAvgDiam = Inputs_dist(List,ipls=dist_bet,Li=dis_start,Lf=dis_end)

    #InfoInp = pd.DataFrame()

    #InfoInp["InpIndx"] = InpSecIndx
    #InfoInp["Loc"] = InpSecloc
    #InfoInp["Dist"] = InpDist
    #InfoInp["InpAvgDiam"] = InpDist
    #InfoInp["Level"] = SecLev[InpSecIndx,1]
    
    InpSecLev = SecLev[InpSecIndx,1]
    
    sig = 2
    
    Dist,Num = np.unique(InpDist,return_counts=True)
    Lev,NumL = np.unique(InpSecLev,return_counts=True)
    Diam,NumL = np.unique(np.round(InpAvgDiam,sig),return_counts=True)
    
    dendInpIndx = []
    dendloc = []
    denddis = []
    dendlev = []
    dendavgdiam = []
    
    for i in range(len(Dist)):
        for j in range(len(Lev)):
            for k in range(len(Diam)):
                
                dd = np.where((InpDist==Dist[i])&(InpSecLev==Lev[j])&(np.round(InpAvgDiam,sig)==Diam[k]))[0]
        
                if len(dd)>=num:

                    indx = np.random.choice(dd,num,replace=False)

                    dendInpIndx.append(InpSecIndx[indx])
                    dendloc.append(InpSecloc[indx])
                    denddis.append(InpDist[indx])
                    dendlev.append(InpSecLev[indx])
                    dendavgdiam.append(InpAvgDiam[indx])

            #elif len(dd)>0:

            #    dendInpIndx.append(InpSecIndx[dd])
            #    dendloc.append(InpSecloc[dd])
            #    denddis.append(InpDist[dd])
            #    dendlev.append(InpSecLev[dd])

    dat = []
    
    dendInpIndx0 = []
    
    for i in range(len(dendInpIndx)):

        for j in range(len(dendInpIndx[i])):
            
            if dendInpIndx[i][j] not in dendInpIndx0:
                
                dendInpIndx0 += dendInpIndx[i][j]
                
                dat.append([dendInpIndx[i][j],dendloc[i][j],denddis[i][j],dendlev[i][j]])

    return dat

