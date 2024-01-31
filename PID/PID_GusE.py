from itertools import combinations

def PID_collec(n):

    ee = [ i for i in range(n+1)]

    Collec = [{}]
    
    for k in range(1,len(ee)):

        Collec += [set(i) for i in combinations(ee[1:len(ee)], k)]

    Siz = np.zeros(len(Collec),dtype=np.int8)
    
    for k in range(len(Collec)):
        
        Siz[k] = len(Collec[k])
        
    return [Collec],Siz
    
def Labels(x,Collec):

    Lab = np.zeros(len(Collec),dtype=np.int8)
    
    for i in range(1,len(Collec)):

        for k in range(len(x)):

            if x[k] == Collec[i]:
    
                Lab[i] = 1

    return Lab

def LabtoCol(Lb,Collec0):

    Collecs = 1*Collec0
    
    for i in range(len(Lb)):
        
        if Lb[i]==0:
            
            Collecs[i] = {}
            
    return Collecs

def RedFuncTest(CollecL,LatEl,Collec0,Siz):

    LB = []
    ln = len(Collec0)

    for m in range(len(CollecL)):
    
        Collec = CollecL[m]

        for j in range(len(Collec)):

            for i in range(1,ln):
    
                CollecS = 1*Collec

                s = 1

                if Collec[i] != {}:

                    #for k in range(i-1,-1,-1):
                    for k in range(0,i,1):
                        
                        if Siz[k]<Siz[i]:
                            
                            for e in list(Collec[i]):

                                if e not in list(Collec[k]):

                                    s = 1*s

                                else:

                                    s = 0*s

                    if s == 1:

                        CollecS[i] = {}

                        lab = [(CollecS[i] != {})*1 for i in range(len(Collec))]

                        if (lab not in LB)and(lab not in LatEl[-1]):

                            LB.append(lab)

    LatEl.append(LB)
    
    CollecS = [LabtoCol(LB[i],Collec0) for i in range(len(LB))]
    
    return LatEl,CollecS

def Atomlab(latel,Collec0):
    
    labC = LabtoCol(latel,Collec0)

    lab0 = []

    i = 1

    print(labC)

    st = labC[-i]

    for e in labC[-1]:

        print(e)

        i = 1

        while i<len(labC):

                for j in range(len(labC)-1,-1,-1):

                    if e in labC[j]:

                        st = labC[j]

                i += 1

        lab0.append(st)

    lab = []

    for i in lab0:

        for j in lab0:

            if i!=j:

                s = 1

                for e in i:

                    if e in j:

                        s = 0

                if s!=0:
                    
                    if i not in lab:
                        lab.append(i)

    return lab

def flatten_concatenation(matrix):
    
    flat_list = []
    
    for row in matrix:
        
        flat_list += row
        
    return flat_list

def ChangeLab(AtomsCode,Collec0,Siz):

    Idx = np.where(np.array(AtomsCode)==1)[0]

    nn = 0

    CC = Collec0*1

    for ic in range(len(AtomsCode)):

        if AtomsCode[ic]==0:
            CC[ic] = {}

    Res = []

    while nn<len(Idx):

        idx = Idx[nn]

        if CC[idx]!={}:

            cc = CC[idx]

            ix = np.where(Siz>=Siz[idx]+1)[0]

            for ii in ix:

                if cc.issubset(CC[ii]):

                    CC[ii] = {}

        nn += 1

    for i in CC:
        if i!={}:
            Res.append(i)

    return Res

def RedVec(sXX):

    B = []
    
    for i in range(len(sXX)):

        xx = str(sXX[i]).split(":")

        xx = ["X_"+xx[i] for i in range(len(xx))]

        red = ""

        if len(xx)>1:

            red = "\min\{"

            for xi in xx[:-1]:
                red += " I("+xi + ',Y) , '

            red += " I("+xx[-1]+',Y) \}'

        else:
            red = "I("+xx[0]+",Y)"
        
        B.append(Symbol(red))
        
    return B

def LabX(sXX):

    B = []

    for nn in range(len(sXX)):

        xx = str(sXX[nn]).split(":")

        xx = ["X_"+xx[i] for i in range(len(xx))]

        lab = "\{"

        if len(xx)>1:
            for xi in xx[:-1]:

                lab += xi + ':'

            lab += xx[-1]+"\}"

        else:
            lab += xx[-1]+"\}"
        
        B.append(Symbol(lab))
        
    return B

def RedLattice(n):

    CollecL,Siz = PID_collec(n)

    Collec0 = CollecL[0]

    lab = [(Collec0[i] != {})*1 for i in range(len(Collec0))]

    LatEl = []

    LatEl.append([lab])

    while (sum(LatEl[-1][0]) > 1):

        LatEl, CollecL = RedFuncTest(CollecL,LatEl,Collec0,Siz)
    
    LatEl0 = []

    for i in range(len(LatEl)):

        for j in range(len(LatEl[i])):

            LatEl0.append(LatEl[i][j])

    Natoms = len(LatEl0)

    Levels = [np.sum(LatEl0[i]) for i in range(len(LatEl0))]

    mS = np.max(Siz)

    for i in range(mS+1):

        LatEl0[0][np.where(Siz)==i]

    XX = [ChangeLab(i,Collec0,Siz) for i in LatEl0]
    
    AA = np.zeros([Natoms,Natoms])

    for i in range(len(LatEl0)):

        ap = LatEl0[i]

        idx = np.where(np.array(Levels)>=Levels[i])[0]

        for j in range(len(idx)):

            a = LatEl0[idx[j]]

            s = 1

            for k in range(len(ap)):

                if ap[k]==1 and a[k]!=ap[k]:

                    s = 0

            if s==1: 
                
                AA[i,idx[j]] = 1
    
    return XX,AA,Collec0
