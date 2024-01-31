using Random,Distributions,CUDA,Combinatorics, CausalityTools, DelimitedFiles

function Com!(ID::Array{Int32,1},ss1::Array{Int32,2},ss2::Array{Int32,2},nn::Array{Int32,1},s::Int32,C::Int32)
    
    nn[1] = 0
    
    for ss in combinations(ID, s)
        
        nn[1] += 1
        
        ss1[nn[1],1:s] = ss
        ss2[nn[1],1:C-s] = ID[ID .∉ Ref(ss)]
        
    end
    
    return nothing
end

function varphiR(X,tau,s1,s2)
    
    MI = IMf(X[1:end-tau,:],X[1+tau:end,:])
    
    XX1 = X[:,s1]
    XX2 = X[:,s2]
    
    H1 = ComplexityMeasures.entropy(probabilities(Dataset(XX1)))
    H2 = ComplexityMeasures.entropy(probabilities(Dataset(XX2)))

    Ka = minimum([H1,H2])
    
    MI1 = IMf(XX1[1:end-tau,:],XX1[1+tau:end,:])
    MIc1 = IMf(XX1[1:end-tau,:],XX2[1+tau:end,:])
    MIc2 = IMf(XX2[1:end-tau,:],XX1[1+tau:end,:])
    MI2 = IMf(XX2[1:end-tau,:],XX2[1+tau:end,:])
    
    if Ka>0
        vphi = (MI - MI1 - MI2 + minimum([MIc1,MIc2]))/Ka
    else
        vphi = (MI - MI1 - MI2 + minimum([MIc1,MIc2]))/1e-5 
    end
    
    return vphi
end

function IMf(x,y)
    
    ps1 = probabilities(Dataset(x));
    
    h1 = ComplexityMeasures.entropy(ps1)

    ps2 = probabilities(Dataset(y));

    h2 = ComplexityMeasures.entropy(ps2)

    ps3 = probabilities(Dataset([x y]));

    h3 = ComplexityMeasures.entropy(ps3)

    return h1+h2-h3
end

function Computs_Caus!(Computs::Array{Float64,1},XX1::Array{Int32,2},XX2::Array{Int32,2},varphiR::Array{Float64,1},MI::Float64,tau::Int32)
    
    Computs[1] = ComplexityMeasures.entropy(probabilities(Dataset(XX1)))
    Computs[2] = ComplexityMeasures.entropy(probabilities(Dataset(XX2)))

    Computs[3] = minimum(Computs[1:2])

    Computs[4] = IMf(XX1[1:end-tau,:],XX1[1+tau:end,:])
    Computs[5] = IMf(XX1[1:end-tau,:],XX2[1+tau:end,:])
    Computs[6] = IMf(XX2[1:end-tau,:],XX1[1+tau:end,:])
    Computs[7] = IMf(XX2[1:end-tau,:],XX2[1+tau:end,:])

    Computs[8] = MI - Computs[4] - Computs[7] + minimum([Computs[5],Computs[6]]) 
    
    if Computs[3]>0
        append!(varphiR,Computs[8]/Computs[3])
    else
        append!(varphiR,Computs[8]/1e-4)
    end
    
    return nothing
end

function Phi_R_parallel!(X::Array{Int32,2},vr::Array{Float32,1},tau::Int32)
    
    allfiles = readdir()

    filesR = [i for i in allfiles if occursin(".txt", i) & occursin("PHIs_", i)]
    
    if length(filesR)>0
        for i in 1:length(filesR)
            rm(filesR[i])
        end
    end
    
    Tt,C = size(X)

    ID = collect(1:C)
    
    # Sizes of bipartition
    Sizes = Array{Int32,2}(reduce(vcat,transpose.(unique([i for i in combinations(append!(ID,ID),2) if (sum(i)==C)&(i[1]-i[2]<=0)]))))
    
    sm = maximum(Sizes[:,1])

    maxS = Int32(binomial(Int32(C),sm))
    
    Ns = Int32(size(Sizes)[1])

    listS = []
    
    Limits = []
    
    for ss in 1:Ns 

        for j in 1:binomial(Int32(C), Int32(ss))

            push!(listS,(ss,j))

        end
    end

    NumProc = size(listS)[1]
    
    Ntr = Threads.nthreads()
    
    NT = Int(ceil(NumProc/Ntr))

    for i in 1:Ntr
        if i*NT < NumProc 
            push!(Limits,((i-1)*NT+1,i*NT)) 
        else i*NT >= NumProc
            push!(Limits,((i-1)*NT+1,NumProc)) 
            break
        end
    end

    ID = Array{Int32,1}(collect(1:C))

    MI = IMf(X[1:end-tau,:],X[1+tau:end,:])

    Ntrs = size(Limits)[1]

    Threads.@threads for i=1:Ntrs

        varphir = zeros(Float64,0)

        Computs = Array{Float64,1}(zeros(8))

        sa = zeros(Int32,1)

        ss1 = Array{Int32,2}(zeros(maxS,C))
        ss2 = Array{Int32,2}(zeros(maxS,C))

        @inbounds for ii in Limits[i][1]:Limits[i][2]

            s = Int32(listS[ii][1])
            j = Int32(listS[ii][2])

            if sa[1]!=s

                nn = zeros(Int32,1)

                Com!(ID,ss1,ss2,nn,s,Int32(C))

                sa .= s

            end

            Computs_Caus!(Computs,X[:,ss1[j,1:s]],X[:,ss2[j,1:(C-s)]],varphir,MI,tau)           

            if Computs[8]==0. 

                break

            end

        end

        vrr = minimum(varphir)

        R = "$(vrr)\n"
	
	filename = "PHIs_$(i)_"

        io = open(filename*".txt", "w+")

        write(io,R)

        close(io)

    end

    GC.gc()

    allfiles = readdir()

    files = [i for i in allfiles if occursin(".txt", i) & occursin("PHIs_", i)]

    vvr = zeros(Float64,0)

    for i in 1:length(files)

        append!(vvr,readdlm(files[i], '\n', Float32));

        rm(files[i])

    end   

    vr[1] = minimum(vvr)
    
    return nothing

end

function MinIJ(X1::Array{Int32},X2::Array{Int32},tau::Int32)
    
    I1 = IMf(X1[1:end-tau,:],X2[tau+1:end,:])
    I2 = IMf(X2[1:end-tau,:],X1[tau+1:end,:])
    
    return minimum([I1,I2])
end

function MinI(X1::Array{Int32},X2::Array{Int32},X::Array{Int32},tau::Int32)
    
    I1 = IMf(X1[1:end-tau,:],X[tau+1:end,:])
    I2 = IMf(X2[1:end-tau,:],X[tau+1:end,:])
    
    return minimum([I1,I2])
end

function MinJ(X1::Array{Int32},X2::Array{Int32},X::Array{Int32},tau::Int32)
    
    I1 = IMf(X[1:end-tau,:],X1[tau+1:end,:])
    I2 = IMf(X[1:end-tau,:],X2[tau+1:end,:])
    
    return minimum([I1,I2])
end

function Compute_Atoms!(X1::Array{Int32},X2::Array{Int32},X::Array{Int32},tau::Int32,B::Array{Float64,1},Atoms::Array{Float64,1})
    
    B[:] = [MinIJ(X1,X2,tau),MinI(X1,X2,X1,tau),MinI(X1,X2,X2,tau),MinI(X1,X2,X,tau),MinJ(X1,X2,X1,tau),
        IMf(X1[1:end-tau,:],X1[1+tau:end,:]),IMf(X1[1:end-tau,:],X2[1+tau:end,:]),IMf(X1[1:end-tau,:],X[1+tau:end,:]),
        MinJ(X1,X2,X2,tau),IMf(X2[1:end-tau,:],X1[1+tau:end,:]),IMf(X2[1:end-tau,:],X2[1+tau:end,:]),IMf(X2[1:end-tau,:],X[1+tau:end,:]),
        MinJ(X1,X2,X,tau),IMf(X[1:end-tau,:],X1[1+tau:end,:]),IMf(X[1:end-tau,:],X2[1+tau:end,:]),IMf(X[1:end-tau,:],X[1+tau:end,:])] 
    
    Atoms[:] = [B[1], -B[1] + B[2], -B[1] + B[3], B[1] - B[2] - B[3] + B[4], -B[1] + B[5], B[1] - B[2] - B[5] + B[6],
        B[1] - B[3] - B[5] + B[7], -B[1] + B[2] + B[3] - B[4] + B[5] - B[6] - B[7] + B[8], -B[1] + B[9], B[1] + B[10] - B[2] - B[9], B[1] + B[11] - B[3] - B[9], -B[1] - B[10] - B[11] + B[12] + B[2] + B[3] - B[4] + B[9], B[1] + B[13] - B[5] - B[9], -B[1] - B[10] - B[13] + B[14] + B[2] + B[5] - B[6] + B[9], -B[1] - B[11] - B[13] + B[15] + B[3] + B[5] - B[7] + B[9], B[1] + B[10] + B[11] - B[12] + B[13] - B[14] - B[15] + B[16] - B[2] - B[3] + B[4] - B[5] + B[6] + B[7] - B[8] - B[9]]    

    return nothing
end

function Phi_R_parallel_Atoms!(X::Array{Int32,2},vr::Array{Float32,1},tau::Int32)
    
    allfiles = readdir()

    filesR = [i for i in allfiles if occursin(".txt", i) & occursin("PHIs_", i)]
    
    if length(filesR)>0
        for i in 1:length(filesR)
            rm(filesR[i])
        end
    end
    
    Tt,C = size(X)

    ID = collect(1:C)
    
    # Sizes of bipartition
    Sizes = Array{Int32,2}(reduce(vcat,transpose.(unique([i for i in combinations(append!(ID,ID),2) if (sum(i)==C)&(i[1]-i[2]<=0)]))))
    
    sm = maximum(Sizes[:,1])

    maxS = Int32(binomial(Int32(C),sm))
    
    Ns = Int32(size(Sizes)[1])

    listS = []
    
    Limits = []
    
    for ss in 1:Ns 

        for j in 1:binomial(Int32(C), Int32(ss))

            push!(listS,(ss,j))

        end
    end

    NumProc = size(listS)[1]
    
    Ntr = Threads.nthreads()
    
    NT = Int(ceil(NumProc/Ntr))

    for i in 1:Ntr
        if i*NT < NumProc 
            push!(Limits,((i-1)*NT+1,i*NT)) 
        else i*NT >= NumProc
            push!(Limits,((i-1)*NT+1,NumProc)) 
            break
        end
    end

    ID = Array{Int32,1}(collect(1:C))

    Ntrs = size(Limits)[1]
    
    varphiAtoms = [4,  8, 12, 13, 14, 15, 16,  7, 10]
    
    Threads.@threads :static for i=1:Ntrs
        
        varphir = zeros(Float64,0)

        K = zeros(Float64,1)

        B = zeros(Float64,16)

        Atoms = zeros(Float64,16);

        sa = zeros(Int32,1)

        ss1 = Array{Int32,2}(zeros(maxS,C))
        ss2 = Array{Int32,2}(zeros(maxS,C))

        @inbounds for ii in Limits[i][1]:Limits[i][2]

            s = Int32(listS[ii][1])
            j = Int32(listS[ii][2])

            if sa[1]!=s

                nn = zeros(Int32,1)

                Com!(ID,ss1,ss2,nn,s,Int32(C))

                sa .= s

            end

            Compute_Atoms!(X[:,ss1[j,1:s]],X[:,ss2[j,1:(C-s)]],X,tau,B,Atoms)

            K[1] = minimum([ComplexityMeasures.entropy(probabilities(Dataset(X[:,ss1[j,1:s]]))),ComplexityMeasures.entropy(probabilities(Dataset(X[:,ss2[j,1:(C-s)]])))])

            if K[1]>=1e-5
                append!(varphir,sum(Atoms[varphiAtoms])/K[1])
            else
                append!(varphir,sum(Atoms[varphiAtoms])/1e-5)
            end
            
            if varphir[end]==0. 

                break

            end

        end

        vrr = minimum(varphir)

        R = "$(vrr)\n"

        filename = "PHIs_$(i)_"

        io = open(filename*".txt", "w+")

        write(io,R)

        close(io)

    end

    GC.gc()

    allfiles = readdir()

    files = [i for i in allfiles if occursin(".txt", i) & occursin("PHIs_", i)]

    vvr = zeros(Float64,0)

    for i in 1:length(files)

        append!(vvr,readdlm(files[i], '\n', Float32));

        rm(files[i])

    end   

    vr[1] = minimum(vvr)
    
    return nothing

end

function PHIID_parallel_Atoms!(X::Array{Int32,2},tau::Int32,fileX::String)
    
    allfiles = readdir()

    filesR = [i for i in allfiles if occursin(".txt", i) & occursin("PHIs_", i)]
    
    if length(filesR)>0
        for i in 1:length(filesR)
            rm(filesR[i])
        end
    end
    
    Tt,C = size(X)

    ID = collect(1:C)
    
    # Sizes of bipartition
    Sizes = Array{Int32,2}(reduce(vcat,transpose.(unique([i for i in combinations(append!(ID,ID),2) if (sum(i)==C)&(i[1]-i[2]<=0)]))))
    
    sm = maximum(Sizes[:,1])

    maxS = Int32(binomial(Int32(C),sm))
    
    Ns = Int32(size(Sizes)[1])

    listS = []
    
    Limits = []
    
    for ss in 1:Ns 

        for j in 1:binomial(Int32(C), Int32(ss))

            push!(listS,(ss,j))

        end
    end

    NumProc = size(listS)[1]
    
    Ntr = Threads.nthreads()
    
    if NumProc/Ntr > 1
    	NT = Int(ceil(NumProc/Ntr))
    else
    	while NumProc/Ntr < 1
    		Ntr -=1
    	end
    	NT = Int(ceil(NumProc/Ntr))
    end
    
    for i in 1:Ntr
        if i*NT < NumProc 
            push!(Limits,((i-1)*NT+1,i*NT)) 
        else i*NT >= NumProc
            push!(Limits,((i-1)*NT+1,NumProc)) 
            break
        end
    end

    ID = Array{Int32,1}(collect(1:C))

    Ntrs = size(Limits)[1]
    
    PhiId_Atoms = -1 .*ones(Float64,NumProc,16)
    Ss = zeros(Int32,NumProc)
    K = zeros(Float64,NumProc,2)

    Threads.@threads :static for i=1:Ntrs

        idthr = Threads.threadid()

        B = zeros(Float64,16)

        Atoms = zeros(Float64,16);

        sa = zeros(Int32,1)

        ss1 = Array{Int32,2}(zeros(maxS,C))
        ss2 = Array{Int32,2}(zeros(maxS,C))

        for ii in Limits[i][1]:Limits[i][2]

            s = Int32(listS[ii][1])
            j = Int32(listS[ii][2])

            if sa[1]!=s

                nn = zeros(Int32,1)

                Com!(ID,ss1,ss2,nn,s,Int32(C))

                sa .= s

            end

            Compute_Atoms!(X[:,ss1[j,1:s]],X[:,ss2[j,1:(C-s)]],X,tau,B,Atoms)

            PhiId_Atoms[ii,:] = Atoms
            Ss[ii] = s
            K[ii,:] = Float64.([ComplexityMeasures.entropy(probabilities(Dataset(X[:,ss1[j,1:s]]))),ComplexityMeasures.entropy(probabilities(Dataset(X[:,ss2[j,1:(C-s)]])))])

        end
    end
    
    io = open(fileX, "w")

    writedlm(io,hcat(PhiId_Atoms,hcat(Ss,K)),",")

    close(io)

    return nothing

end

function IMc(x::Array{Float32},y::Array{Float32})
    
    I = mutualinfo(MIShannon(base = ℯ),Kraskov(k = 5),Dataset(x), Dataset(y))
    
    return I
    
end

function MinIJ_c(X1p::Array{Float32},X1f::Array{Float32},X2p::Array{Float32},X2f::Array{Float32})

    I1 = mutualinfo(MIShannon(base = ℯ),Kraskov(k = 5),Dataset(X1p), Dataset(X2f))
    I2 = mutualinfo(MIShannon(base = ℯ),Kraskov(k = 5),Dataset(X2p), Dataset(X1f))
    
    return minimum([I1,I2])
end

function MinI_c(X1p::Array{Float32},X2p::Array{Float32},Xf::Array{Float32})
    
    I1 = mutualinfo(MIShannon(base = ℯ),Kraskov(k = 5),Dataset(X1p), Dataset(Xf))
    I2 = mutualinfo(MIShannon(base = ℯ),Kraskov(k = 5),Dataset(X2p), Dataset(Xf))
        
    return minimum([I1,I2])
end

function MinJ_c(X1f::Array{Float32},X2f::Array{Float32},Xp::Array{Float32})
    
    I1 = mutualinfo(MIShannon(base = ℯ),Kraskov(k = 5),Dataset(Xp), Dataset(X1f))
    I2 = mutualinfo(MIShannon(base = ℯ),Kraskov(k = 5),Dataset(Xp), Dataset(X2f))
    
    return minimum([I1,I2])
end

function Compute_Atoms_cont!(X1::Array{Float32},X2::Array{Float32},X::Array{Float32},tau::Int32,B::Array{Float64,1},Atoms::Array{Float64,1})
    
    X1p = X1[1:end-tau,:]
    X1f = X1[1+tau:end,:]
    
    X2p = X2[1:end-tau,:]
    X2f = X2[1+tau:end,:]
    
    Xp = X[1:end-tau,:]
    Xf = X[1+tau:end,:]
    
    B[:] = [MinIJ_c(X1p,X1f,X2p,X2f),MinI_c(X1p,X2p,X1f),MinI_c(X1p,X2p,X2f),MinI_c(X1p,X2p,Xf),MinJ_c(X1f,X2f,X1p),
        IMc(X1p,X1f),IMc(X1p,X2f),IMc(X1p,Xf),MinJ_c(X1f,X2f,X2p),IMc(X2p,X1f),IMc(X2p,X2f),IMc(X2p,Xf), MinJ_c(X1p,X2p,Xf),IMc(Xp,X1f),IMc(Xp,X2f),IMc(Xp,Xf)] 
    
    Atoms[:] = [B[1], -B[1] + B[2], -B[1] + B[3], B[1] - B[2] - B[3] + B[4], -B[1] + B[5], B[1] - B[2] - B[5] + B[6],
        B[1] - B[3] - B[5] + B[7], -B[1] + B[2] + B[3] - B[4] + B[5] - B[6] - B[7] + B[8], -B[1] + B[9], B[1] + B[10] - B[2] - B[9], B[1] + B[11] - B[3] - B[9], -B[1] - B[10] - B[11] + B[12] + B[2] + B[3] - B[4] + B[9], B[1] + B[13] - B[5] - B[9], -B[1] - B[10] - B[13] + B[14] + B[2] + B[5] - B[6] + B[9], -B[1] - B[11] - B[13] + B[15] + B[3] + B[5] - B[7] + B[9], B[1] + B[10] + B[11] - B[12] + B[13] - B[14] - B[15] + B[16] - B[2] - B[3] + B[4] - B[5] + B[6] + B[7] - B[8] - B[9]]    
    
    return nothing
end

function PHIID_parallel_Atoms_cont!(X::Array{Float32,2},tau::Int32,fileX::String)
        
    Tt,C = size(X)

    ID = collect(1:C)
    
    # Sizes of bipartition
    Sizes = Array{Int32,2}(reduce(vcat,transpose.(unique([i for i in combinations(append!(ID,ID),2) if (sum(i)==C)&(i[1]-i[2]<=0)]))))
    
    sm = maximum(Sizes[:,1])

    maxS = Int32(binomial(Int32(C),sm))
    
    Ns = Int32(size(Sizes)[1])

    listS = []
    
    Limits = []
    
    for ss in 1:Ns 

        for j in 1:binomial(Int32(C), Int32(ss))

            push!(listS,(ss,j))

        end
    end

    NumProc = size(listS)[1]
    
    Ntr = Threads.nthreads()
    
    if NumProc/Ntr > 1
    	NT = Int(ceil(NumProc/Ntr))
    else
    	while NumProc/Ntr < 1
    		Ntr -=1
    	end
    	NT = Int(ceil(NumProc/Ntr))
    end
    
    for i in 1:Ntr
        if i*NT < NumProc 
            push!(Limits,((i-1)*NT+1,i*NT)) 
        else i*NT >= NumProc
            push!(Limits,((i-1)*NT+1,NumProc)) 
            break
        end
    end

    ID = Array{Int32,1}(collect(1:C))

    Ntrs = size(Limits)[1]
    PhiId_Atoms = -1 .*ones(Float64,NumProc,16)
    Ss = zeros(Int32,NumProc)
    K = zeros(Float64,NumProc,2)
    
    Threads.@threads :static for i=1:Ntrs

        B = zeros(Float64,16)
        Atoms = zeros(Float64,16);

        sa = zeros(Int32,1)

        ss1 = Array{Int32,2}(zeros(maxS,C))
        ss2 = Array{Int32,2}(zeros(maxS,C))

        for ii in Limits[i][1]:Limits[i][2]

            s = Int32(listS[ii][1])
            j = Int32(listS[ii][2])

            if sa[1]!=s

                nn = zeros(Int32,1)

                Com!(ID,ss1,ss2,nn,s,Int32(C))

                sa .= s

            end

            Compute_Atoms_cont!(X[:,ss1[j,1:s]],X[:,ss2[j,1:(C-s)]],X,tau,B,Atoms)

            PhiId_Atoms[ii,:] = Atoms
            Ss[ii] = s
            K[ii,:] = Float64.([ComplexityMeasures.entropy(Kraskov(k=5), Dataset(X[:,ss1[j,1:s]])),ComplexityMeasures.entropy(Kraskov(k=5), Dataset(X[:,ss2[j,1:(C-s)]]))])

        end
    end
	
    io = open(fileX, "w")

    writedlm(io,hcat(PhiId_Atoms,hcat(Ss,K)),",")

    close(io)
    
    GC.gc(true)
	    
    return nothing

end

#FUNTIONS TO RUN ON GPU
function Entropy_CUDA(Xd,sts,Hd)
    
    T = size(Xd)[1]
    
    id = (blockIdx().x-1)*blockDim().x+threadIdx().x
    stridex = blockDim().x * gridDim().x
    
    d = size(sts)[1]
    
    for i = id:stridex:d
        
        n = Float64(0)
    
        sta = sts[i]
        
        for t in 1:T
            
            std = Xd[t]
            
            if std==sta
                
                n += 1.
            
            end
        end
        
        @inbounds Hd[i] += -(n/T)*log2(n/T)
            
    end

    return nothing

end

function MutualInf_CUDA(Xd,Yd,sts1,sts2,MHd)
    
    T = size(Xd)[1]
    
    id = (blockIdx().x-1)*blockDim().x+threadIdx().x
    stridex = blockDim().x * gridDim().x
    
    yi = blockIdx().y
    
    ss1 = sts1[yi]
    
    d = size(sts2)[1]
    
    n = Float64(0)
    
    for i = id:stridex:d
    
        ss2 = sts2[i]
        
        for t in 1:T
            
            st1 = Xd[t]
            st2 = Yd[t]
        
            if (ss1==st1)&(ss2==st2)
                n += 1.
            end
            
        end

        if n!=0
            @inbounds MHd[yi + d*(i-1)] = -(n/T)*log2(n/T)
        else
            @inbounds MHd[yi + d*(i-1)] = 0.
        end 
        
    end
    
    return nothing

end

function MutualInfor_CUDA(Xs,Ys,Xx,Yy)

    Xd = CuArray(Xs)
    Yd = CuArray(Ys)
    Xxd = CuArray(Xx)
    Yyd = CuArray(Yy)

    sk1 = size(Xx)[1]
    sk2 = size(Yy)[1]

    Tmax = 512

    numblocksxk = ceil(Int, sk1/Tmax)

    H1d = CUDA.zeros(Float64,sk1)
    H2d = CUDA.zeros(Float64,sk2)

    CUDA.@sync begin
        @cuda threads=Tmax blocks=(numblocksxk) Entropy_CUDA(Xd,Xxd,H1d)
    end

    numblocksxk = ceil(Int, sk2/Tmax)

    CUDA.@sync begin
        @cuda threads=Tmax blocks=(numblocksxk) Entropy_CUDA(Yd,Yyd,H2d)
    end

    numblocksxk2 = ceil(Int, sk2/Tmax)

    #To computing mutualinformation by partitions, number of possible states
    MHd = CUDA.zeros(Float64,sk1*sk2)

    CUDA.@sync begin
        @cuda threads=Tmax blocks=(numblocksxk2,sk1) MutualInf_CUDA(Xd,Yd,Xxd,Yyd,MHd)
    end

    H1d = reduce(+,H1d)
    H2d = reduce(+,H2d)

    MI = H1d .+ H2d .- reduce(+,MHd)
   
    return MI
end

#FUNTIONS TO RUN ON GPU
function Partitions_States(Xd,M1d,ss1d)
    
    d1 = size(ss1d)[1]
    
    id = (blockIdx().x-1)*blockDim().x+threadIdx().x
    stridex = blockDim().x * gridDim().x
    
    j = blockIdx().y
    k = blockIdx().z
    
    di = size(Xd)[1]
    
    for i = id:stridex:di
        
        sx1 = ss1d[k,j]
        
        @inbounds M1d[i,k,j] = Xd[i,sx1]*(2^(j-1))
                    
    end

    return nothing

end

#FUNTIONS TO RUN ON GPU
function Partitions_Entropy(Mds,Hd,sts)
    
    d1,d2 = size(Mds)
    
    id = (blockIdx().x-1)*blockDim().x+threadIdx().x
    stridex = blockDim().x * gridDim().x
    
    for i = id:stridex:d2
        nn = 1
    
        n = Float64(1.)
    
        sta = Mds[1,i]
        
        sts[i,1] = sta
                
        for xi in 1:d1
            
            std = Mds[xi,i]
            
            if std==sta
                
                n += 1.
            
            else
                nn += 1
                
                sts[i,nn] = std
                
                sta = 1*std
                
                @inbounds Hd[i] += -(n/d1)*log2(n/d1)
                
                n = Float64(1.)
                
            end
            
        end
        
        if (n!=1. *d1)
            @inbounds Hd[i] += -(n/d1)*log2(n/d1)
        else
            @inbounds Hd[i] += 0.
        end    
    end

    return nothing

end

function Entropy_Part(Md,sk,States=true)
    
    Mds = sort(Md,dims=1)
    
    T,sl = size(Mds)
    
    Hd = CUDA.zeros(Float64,sl)
    
    sts = CUDA.ones(Int,sl,sk).*-1
    
    CUDA.@sync begin
        @cuda threads=ThrNum blocks=(numblocksx) Partitions_Entropy(Mds,Hd,sts)
    end
    
    if States==true
        return Hd,sts
    else
        return Hd
    end
end

#FUNTIONS TO RUN ON GPU
function Partitions_MutualInf(M1d,M2d,sts1,sts2,MHd)
    
    d1,d2 = size(M1d)
    
    id = (blockIdx().x-1)*blockDim().x+threadIdx().x
    stridex = blockDim().x * gridDim().x
    
    dxi = blockIdx().y
    dyi = blockIdx().z
    
    Ny = gridDim().y
    
    for i = id:stridex:d2
        ss1 = sts1[i,dxi]
        ss2 = sts2[i,dyi]
        
        n = Float64(0)
            
        for j in 1:d1

            st1 = M1d[j,i]
            st2 = M2d[j,i]

            if (ss1==st1)&(ss2==st2)
                n += 1.
            end

        end
        
        if (n!=0.)&(n!= 1. *d1)
            @inbounds MHd[i,dxi + Ny*(dyi-1)] = -(n/d1)*log2(n/d1)
        else
            @inbounds MHd[i,dxi + Ny*(dyi-1)] = 0.    
        end
    end

    return nothing

end

function MutualInf_Part(M1d,M2d,sts1,sts2,tau)
    
    sk1 = size(sts1)[2]
    sk2 = size(sts2)[2]
    
    #To computing mutualinformation by partitions, number of possible states
    MHd = CUDA.zeros(Float64,N1,sk1*sk2)

    CUDA.@sync begin
        @cuda threads=ThrNum blocks=(numblocksx,sk1,sk2) Partitions_MutualInf(M1d[1:end+1-tau,:],M2d[tau:end,:],sts1,sts2,MHd)
    end
    
    H1 = Entropy_Part(M1d[1:end+1-tau,:],sk1,false)
    H2 = Entropy_Part(M2d[tau:end,:],sk2,false)
    
    return H1.+ H2 .- reduce(+,MHd,dims=2)
end

function VarPhi_R_CUDA(X::Array{Int32,2},tau::Int32)

    Xd = CuArray(X)

    T,C = size(X)

    ID = collect(1:C)

    # Sizes of bipartition
    Sizes = unique([i for i in combinations(append!(ID,ID),2) if (sum(i)==C)&(i[1]-i[2]<=0)])

    ID = collect(1:C)

    varphiR = zeros(0)

    Xs = ListStates(X[1:end+1-tau,:])
    Ys = ListStates(X[tau:end,:])

    Xx = collect(Set(Xs))
    Yy = collect(Set(Ys));

    MI = MutualInfor_CUDA(Xs,Ys,Xx,Yy)

    global ThrNum
    global numblocksx
    global N1
    
    Tmax = 512
    
    numblocksxT = ceil(Int, T/Tmax)

    for s in 1:length(Sizes)

        ss1 = collect(combinations(ID, Sizes[s][1]))
        ss2 = [ID[ID .∉ Ref(ss1[i])] for i in 1:length(ss1)];

        ss1d = CuArray(reduce(vcat,transpose.(ss1)))
        ss2d = CuArray(reduce(vcat,transpose.(ss2)))

        sl1, sk1 = size(ss1d)
        sl2, sk2 = size(ss2d)

        #############################################
        # GET THINGS READY TO GPU
        #############################################
        N1 = length(ss1)

        if N1<=Tmax
            ThrNum = N1
        else
            ThrNum = Tmax
        end

        numblocksx = ceil(Int, N1/ThrNum)

        M1d = CUDA.zeros(Int,T,sl1,sk1)
        M2d = CUDA.zeros(Int,T,sl2,sk2)

        CUDA.@sync begin
            @cuda threads=ThrNum blocks=(numblocksxT,sk1,sl1) Partitions_States(Xd,M1d,ss1d)
        end

        CUDA.@sync begin
            @cuda threads=ThrNum blocks=(numblocksxT,sk2,sl2) Partitions_States(Xd,M2d,ss2d)
        end

        M1d = reduce(+,M1d,dims=3)[:,:]
        M2d = reduce(+,M2d,dims=3)[:,:];

        H1d,sts1 = Entropy_Part(M1d,2^sk1)
        H2d,sts2 = Entropy_Part(M2d,2^sk2)

        sn1 = maximum(reduce(+,(sts1.>-1),dims=2))
        sts1 = sts1[:,1:sn1] 

        sn2 = maximum(reduce(+,(sts2.>-1),dims=2))
        sts2 = sts2[:,1:sn2] 

        MI1 = MutualInf_Part(M1d,M1d,sts1,sts1,tau)
        MI2 = MutualInf_Part(M1d,M2d,sts1,sts2,tau)
        MI3 = MutualInf_Part(M2d,M1d,sts2,sts1,tau)
        MI4 = MutualInf_Part(M2d,M2d,sts2,sts2,tau)
        
        Kbeta = Array(minimum(mapreduce(permutedims, vcat, [H1d,H2d]),dims=1)')
        
        varR = Array(MI .- MI1 .- MI4 .+ minimum(mapreduce(permutedims, vcat, [MI1,MI2,MI3,MI4]),dims=1)')
        
        varR = varR.*(varR.>=1e-4)
        
        for i in 1:length(varR)
            if Kbeta[i] >= 1e-5
                append!(varphiR,varR[i]/Kbeta[i])
            else
                append!(varphiR,varR[i])#/1e-4)
            end
        end
    end
    return minimum(varphiR)
end
