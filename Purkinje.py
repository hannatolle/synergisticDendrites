'''* Purkinje19b972 current clamp simulation: sample traces of somatic  */
/* and dendritic epsps                                                */
/* synapse at "dend2" (dendA1_001011(1))                              */
/* Michael Hausser and Arnd Roth                                      */
/* Version 1.2 for LASCON 2014                              14.1.2014 */
'''
from neuron import h,gui
from matplotlib import pyplot as plt

from neuron.units import ms, mV

# initial parameters 
t = 0                   # simulation starts at t = 0 ms */
dt = 0.01               # time step, ms */
Vrest = -70             # resting potential, mV */

h.load_file("Purkinje19b972-1.nrn")  # load the morphology file */

shape_window = h.Shape()
shape_window.exec_menu('Show Diam')

# membrane properties are defined here */
membranecap = 0.638856    # specific membrane capacitance in uF cm^-2 */
membraneresist = 120236.0 # specific membrane resistance in ohm cm^2 */
axialresist = 141.949     # axial resistivity in ohm cm */

# all sections
allsec = h.allsec()
for sec in h.allsec():
	sec.insert('pas')
	sec.e_pas=Vrest

# dend sections
dend = [s for s in allsec if s.name().startswith('dend')]
for sec in dend:
	sec.g_pas = 5.34/membraneresist
	sec.Ra = axialresist
	sec.cm = 5.34*membranecap

# dendA1_0 sections (spiny)
dendA1_list = ['dendA1_0', 'dendA1_00', 'dendA1_001', 'dendA1_0010', 'dendA1_00101', 'dendA1_001011', 
	'dendA1_0010110', 'dendA1_0010111', 'dendA1_00101110', 'dendA1_001011101', 'dendA1_00101111', 'dendA1_001011110' 
	'dendA1_0010111101',  'dendA1_00101111011', 'dendA1_0011', 'dendA1_00110', 'dendA1_001101', 'dendA1_0011010', 
	'dendA1_0011011', 'dendA1_00110110', 'dendA1_01', 'dendA1_010', 'dendA1_011', 'dendA1_0100', 'dendA1_0101'
	'dendA1_01001', 'dendA1_010010', 'dendA1_0100100', 'dendA1_01001001', 'dendA1_010010010']

dendA1 = [s for s in allsec if s.name() in dendA1_list]
for sec in dendA1:
	sec.g_pas = 1.2/membraneresist
	sec.Ra = axialresist
	sec.cm = 1.2*membranecap

# soma sections
soma = [s for s in allsec if s.name().startswith('soma')]
for sec in soma:
	sec.g_pas = 1.0/membraneresist
	sec.Ra = axialresist
	sec.cm = 1.0*membranecap

# axon sections
axon = [s for s in allsec if s.name().startswith('axon')]
for sec in axon:
	sec.g_pas = 1.0/membraneresist
	sec.Ra = axialresist
	sec.cm = 1.0*membranecap

# some axonA1 sections
axonA1_list = ['axonA1_0', 'axonA1_000', 'axonA1_0000', 'axonA1_0001', 'axonA1_01', 'axonA1_010']
axonA1 = [s for s in allsec if s.name() in axonA1_list]
for sec in axonA1:
	sec.g_pas = 0.1/membraneresist
	sec.Ra = axialresist
	sec.cm = 0.1*membranecap

# current clamp dend
#stim = h.IClamp(h.dendA1_01001(0.7))
#stim.dur = 0.5
#stim.amp = 1

# current clamp somaA
#stim2 = h.IClamp(h.somaA(0.5))
#stim2.dur = 0.5
#stim2.amp = 1

ISI = 1 * ms
NUM = 20
START = 1 * ms
NOISE = True

TSTOP = 100 * ms

syn_ = h.Exp2Syn(h.dendA1_0(0.5))
syn_.tau1 = 4* ms
syn_.tau2 = 10* ms
syn_.e = 0*mV

ns = h.NetStim()

ns.interval = ISI
ns.number = NUM
ns.start = START
ns.noise = NOISE

# specify the (i, 0, 0)th random stream
ns.noiseFromRandom123(0, 0, 0)

ns.seed(1)

nc = h.NetCon(ns,syn_)

nc.delay = 1
nc.weight[0] = 0.4

# record soma voltage and time
t_vec = h.Vector()

# Set recording vectors
syn_i_vec = h.Vector()
syn_i_vec.record(syn_._ref_i)

# Show stim on 3d plot
shape_window.point_mark(syn_,0)

#shape_window.point_mark(ns,3)

v_vec_soma = h.Vector()
#v_vec_soma.record(h.somaA(0.5)._ref_v) # change recoding pos

v_vec_soma.record(h.dendA1_01001(0.5)._ref_v)

#v_vec_dend = h.Vector()
#v_vec_dend.record(h.dendA1_01001(0.7)._ref_v)
t_vec.record(h._ref_t)

# run simulation
h.tstop = TSTOP  # ms
h.v_init = Vrest
h.run()  
    
# plot voltage vs time
plt.figure(figsize=(8,4)) # Default figsize is (8,6)
plt.plot(t_vec, v_vec_soma, 'r', label='soma')
plt.plot(t_vec, syn_i_vec, 'b', label='syn')
plt.xlabel('time (ms)')
plt.ylabel('mV')
plt.legend()
#plt.ylim([-70,-40])
plt.show()