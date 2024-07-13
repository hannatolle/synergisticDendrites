using DelimitedFiles
using Random,Distributions,CausalityTools, DelimitedFiles

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

#include("PID.jl");

ddir = "Raw_data/"

allfiles = readdir(ddir)

filesR = [i for i in allfiles if occursin(".csv", i)&occursin("Potential_values_", i)]

ddirD = ddir*split(filesR[ni],".c")[1]*"_Continous"

isdir(ddirD) || mkdir(ddirD)

V = readdlm(ddir*filesR[ni],',',Float32,skipstart=1);

est = Kraskov(k = 10, base = ℯ)

Vs = V[:,:];

dt = 0.125

Tau_c = Int.(collect(Int.(round.(LinRange(1,20,20),digits=0)))./dt);

T = size(Vs)[1]

N = size(Vs)[2]-1

hT = ComplexityMeasures.entropy(est,Vs[:,1])

io = open(ddirD*"/Target_entropy_timebin.txt","w")

writedlm(io,hT,",")

close(io)

Threads.@threads :static for t=1:20

    Res = zeros(Float32,N,N,4)

    for i in 1:N-1

        for j in i+1:N

            Res[i,j,:] = PID_c(Vs[1+Tau_c[t]:end,1],Vs[1:end-Tau_c[t],[i+1,j+1]])
            Res[j,i,:] = Res[i,j,[1,3,2,4]]

        end

    end

    PIDlab = ["Red","Un1","Un2","Syn"]

    ii = [1,2,3,4]

    for i in 1:length(PIDlab)

        io = open(ddirD*"/PID_"*PIDlab[i]*"_tau_$(Tau_c[tk]).txt","w")

        writedlm(io,Res[:,:,ii[i]]./hT,",")

        close(io)

    end

end
