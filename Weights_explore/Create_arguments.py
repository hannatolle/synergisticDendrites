import numpy as np

Cases = ["Both","AMPA","NMDA"]

dw = 1e-5

WW = np.arange(2e-4,4e-4+dw,dw)

f = open("arguments.txt", "w")

for Case in Cases:
	for ww in WW:
		f.write(Case+" "+str(np.round(ww,5))+"\n")
    #f.write(Case+"\n")

f.close()
