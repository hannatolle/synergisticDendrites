from neuron import h

##################################################
#Initialize Neuron model
ddir = "L5bPCmodelsEH/"

h.load_file("import3d.hoc")
h.load_file("stdgui.hoc")
h.load_file(ddir+"models/L5PCbiophys3.hoc")
h.load_file(ddir+"models/L5PCtemplate.hoc")

morphology_file = ddir+"morphologies/cell1.asc"

cell = h.L5PCtemplate(morphology_file)
####################################################
#Dendritic sections labels
Sections = [i for i in h.allsec() if ("axon" not in str(i))and("soma" not in str(i))]

#Number of sections
N = len(Sections)

f = open("arguments.txt", "w")

for i in range(N):
    for j in range(N-1):
        f.write(str(i)+" "+str(j)+"\n")

f.close()
