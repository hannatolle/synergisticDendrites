using DelimitedFiles
using Random,Distributions,CausalityTools, DelimitedFiles

function IMf(x,y)
    
    ps1 = probabilities(Dataset(x));
    
    h1 = ComplexityMeasures.entropy(Shannon(; base = 2),ps1)

    ps2 = probabilities(Dataset(y));

    h2 = ComplexityMeasures.entropy(Shannon(; base = 2),ps2)

    ps3 = probabilities(Dataset([x y]));

    h3 = ComplexityMeasures.entropy(Shannon(; base = 2),ps3)

    return h1+h2-h3
end

function RedSinIndex(S,Y)
    
    Ss = IMf(S,Y)
    
    nt = size(S)[2]
    
    RSI = 1*Ss
    
    for i in 1:nt
       
        RSI -=  IMf(S[:,i],Y)
        
    end
    
    return RSI
end

function PID(T,X)
    
    RR = minimum([IMf(T,X[:,1]),IMf(T,X[:,2])])
    
    ru1 = IMf(X[:,1],T)
    ru2 = IMf(X[:,2],T)
        
    U1 = ru1 - RR

    U2 = ru2 - RR
    
    Syn = IMf(X,T) - ru1 - ru2 + RR
        
    return [RR,U1,U2,Syn]
end

function IMf_c(x,y)

    return mutualinfo(MIShannon(base = ℯ),Kraskov(k = 10),Dataset(x), Dataset(y))

end

function PID_c(T,X)

    RR = minimum([IMf_c(T,X[:,1]),IMf_c(T,X[:,2])])
    
    ru1 = IMf_c(X[:,1],T)
    ru2 = IMf_c(X[:,2],T)
        
    U1 = ru1 - RR

    U2 = ru2 - RR
    
    Syn = IMf_c(X,T) - ru1 - ru2 + RR
        
    return [RR,U1,U2,Syn]
end