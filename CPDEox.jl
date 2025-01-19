module CPDE
export init_par, wrap_F,wrap_Jacobian, meanval 
using LinearAlgebra, SparseArrays
function meanval(z)
    y=0.5*(z[1:end-1]+z[2:end])
    return y
end
function COX(z,n,par)
    zz=collect(z)
    COXmax=par["max_consumption"]
    hz=par["hz"]
  #  phencons= COXmax*((zz.*(1.0 .-zz))/0.25 .+1.0)/2.0

    totCOX=COXmax*hz*sum(n)
    
    return totCOX
end
function prol(z,rho,uox,par)

    cH=par["cH"]
    cmax=par["DC"]["pmax"]
    cHS=0.05
    cmaxs=par["CSC"]["pmax"]
    V_max=1 #maximum capacity

    fz=uox.^4/(cH^4+uox.^4).*exp(-(z-0.55).^2/0.04);
    fz2=uox^4/(cHS^4+uox.^4).*exp(-z.^2/0.01);

    y =(1-rho/V_max)*(cmax*fz+cmaxs*fz2);
    return y
end
function dprol(z,rho,uox,par)
    cH=par["cH"]
    cmax=par["DC"]["pmax"]
    cHS=0.05
    cmaxs=par["CSC"]["pmax"]

    fz=cH^4/(cH^4+uox^4)^2*exp(-(z-0.55)^2/0.04);
    fz2=cHS^4/(cHS^4+uox^4)^2*exp(-z^2/0.01);

    y =(1-rho)*(cmax*fz+cmaxs*fz2)*4*uox^3;
    return y
end
function death(z,uox,par)

    dAp=par["dAp"]
    dN=par["dN"]
    cN=par["cN"]
    kap = 10;
    #death by apoptosis
    epsilon=0.01;
    threshold=(tanh((uox-cN)/epsilon)+1)/2;
    
    y = dAp*(exp(-kap*abs(z-1)))+dN*(1-threshold);
    return y
end
function ddeath(z,uox,par)
   
    dN=par["dN"]
    cN=par["cN"]
    epsilon=0.01;
    dth=1/2/epsilon/cosh((uox -cN)/epsilon)^2
    return -dN*dth
end
function velocity(uox,par)
    cH=par["cH"]
    Vp=par["Vp"]*par["magnVp"];
    Vm=par["Vm"];
    epsilon=0.05
    hyp_treshold=(tanh((uox-cH)/epsilon)+1)/2
    y=Vp*hyp_treshold-Vm*(1-hyp_treshold)

    return y
end
function dvel(uox,par)

    cH=par["cH"]
    Vp=par["Vp"]*par["magnVp"];
    Vm=par["Vm"]
    epsilon=0.05
    dth=1/2/cosh((uox-cH)/epsilon)^2/epsilon
    y=(Vp+Vm)*dth

    return y    
end
function Adv_OP(uox,par)
    
    nz=par["nz"]
    vel=velocity(uox,par);
   
    
    fluxm=spdiagm(1 => ones(nz-1),0 => ones(nz))
    fluxp=spdiagm(-1 => ones(nz-1),0 => ones(nz))
    
    fluxp[1,:].=0

    fluxm[end,:].=0
    
    T=(-vel[1:end-1].*fluxp+vel[2:end].*fluxm)/par["hz"]/2
    return T
end

function dNdc(Uold,uox,spz,par)
    nz=par["nz"]

    fluxm=spdiagm(1 => ones(nz-1),0 => ones(nz))
    fluxp=spdiagm(-1 => ones(nz-1),0 => ones(nz))
    
    fluxp[1,:].=0

    fluxm[end,:].=0
    
    dv=dvel(uox,par)
    T=(-dv[1:end-1].*fluxp+dv[2:end].*fluxm)/par["hz"]/2
    
    transport=T*Uold;
    int_u=par["hz"]*sum(Uold);
    dF=dprol.(meanval(spz),int_u,uox,Ref(par))-ddeath.(meanval(spz),uox,Ref(par));    
    return -transport+dF.*Uold
end
function nonlocal_OP(Uold,uox,zz,par)
    prolif=prol.(zz,0,uox,Ref(par)) 
    Op=zeros(par["nz"],par["nz"])
    
    for i in 1:size(Op)[1]
        Op[i,:].=prolif[i]*par["hz"]*Uold[i]
    end
    return Op
end

function wrap_Jacobian(Uold,uox,OpDzz,OpDxx,OpOx,spz,par)
    
    M=(par["nx"]+1)*(par["nz"]+1)-1
    Op=zeros(M,M)
    gamma=zeros((length(uox)+1,1))

    mZ=meanval(spz)
    COXmax=par["max_consumption"]
    phencons= COXmax#*((mZ.*(2 .-mZ)).+1)/2
    indc=par["nz"]*(1+par["nx"])
    for k in 1:(par["nx"]+1)
        ind_0=(k-1)*(par["nz"])+1
        ind_f=ind_0+par["nz"]-1
        int_u=par["hz"]*sum(Uold[:,k]);
        if k<par["nx"]+1
            cox=uox[k]   
        else
            cox=1.0
        end 
        
        tempA=prol.(mZ,int_u,cox,Ref(par))-death.(mZ,cox,Ref(par))
        
        tempB=spdiagm(0=>tempA); 
        
        entry=-Adv_OP(cox,par)-nonlocal_OP(Uold[:,k],cox,mZ,par)+tempB+OpDzz 
        Op[ind_0:ind_f,ind_0:ind_f].=entry   
        
        if k<par["nx"]+1
            Op[ind_0:ind_f,indc+k].=dNdc(Uold[:,k],cox,spz,par)   
            Op[indc+k,ind_0:ind_f].=-par["hz"]*(1+tanh((cox-par["cN"])/0.05))/2*phencons
            gamma[k]=COX(meanval(spz),Uold[:,k], par)*1/2/0.05/cosh((cox-par["cN"])/0.05)^2
        end
    end 

    for k in 1:par["nz"]
        Op[k:par["nz"]:indc,k:par["nz"]:indc]+=OpDxx; 
    end
    Op[indc+1:end,indc+1:end]+=(OpOx[1:end-1,1:end-1]-spdiagm(0=>gamma[1:end-1]))

    return Op
    
end  

function wrap_F(N,uox,OpDzz,OpDxx,Opox,spz,par)
    

    zcomp=zeros(size(N))
    xcomp=zeros(size(N))
    gamma=zeros((length(uox)+1,1))
    nx=par["nx"]
    nz=par["nz"]
    

    append!(uox,[1.0])
    F=zeros((nx+1)*(nz+1)-1)
    for j in 1:size(N)[2]
        cox=uox[j]
        zcomp[:,j].=dyz(N[:,j],cox,OpDzz,spz,par)
        gamma[j]=COX(meanval(spz), N[:,j],par) 
    end
    for j in 1:size(N)[1]
        v=vcat(N[j,:]...)
        xcomp[j,:].=OpDxx*v
    end
    
    temp=xcomp+zcomp
    FC=Opox[1:end-1,:]*uox-gamma[1:end-1].*(1.0 .+tanh.((uox[1:end-1].-par["cN"])/0.05))/2
    F[1:end-nx]=vec(temp)
    F[end-nx+1:end]=FC
    return F
end

function dyz(u,uox,D2z,spz,par)

    meanspz=meanval(spz);
    
    u=vcat(u...)
    T=Adv_OP(uox,par)
    transport=T*u
    Duzz=D2z*u;
    int_u=par["hz"]*sum(u);
    prolif=prol.(meanspz,int_u,uox,Ref(par));    
    net_prolif=prolif-death.(meanspz,uox,Ref(par))
    return Duzz-transport+net_prolif.*u
end



function init_par(nx,nz)
    par=Dict()
    # number of lattice sites.
    par["nx"]=nx
    par["nz"]=nz
    # spacing of the lattice.
    hx=1/nx; hz=1/nz;
    par["hx"]=hx
    par["hz"]=hz
    # spatial variable
    spx=range(0,stop=nx,length=nx+1)*hx;
    par["spx"]=spx
    # stemness variable
    spz=range(0,stop=nz,length=nz+1)*hz;
    par["spz"]=spz

    # proliferation parameters 
    cmax=0.02;
    par["DC"]=Dict()
    par["CSC"]=Dict()
    par["DC"]["pmax"]=cmax
    par["CSC"]["pmax"]=5e-3

    par["dAp"]=0.001
    par["dN"]=0.1
    par["Dox"] = 6.3; 
    par["Dx"]=1e-4;
    par["cH"] = 0.3; 
    par["max_consumption"]=10;
    par["cN"]=0.0125;
    par["cinf"]=1;
    # velocity parameters
    epsp=0.1
    omegap=1
    par["magnVp"]=5.5e-4
    par["omegap"]=omegap
    par["epsp"]=epsp
    pos_velocity=[tanh(zz^omegap/epsp)*tanh((1-zz)/epsp) for zz in spz];
    V=pos_velocity/maximum(pos_velocity); #normalized velocity
    par["Vp"]=V
    epsm=0.1
    omegam=2
    neg_vel=[tanh(zz/epsm)*tanh((1-zz)^omegam/epsm) for zz in spz];
    par["magnVm"]=2e-4;
    par["omegam"]=omegam;
    par["epsm"]=epsm
    V=par["magnVm"]*neg_vel/maximum(neg_vel);
    par["Vm"]=V
    par["Dz"]=5e-6;      
    OpDzz,OpDxx,OpOx=init_diff(par,nx,nz)
    return par,OpDzz,OpDxx,OpOx
end

function init_diff(par,nx,nz)
        hx=par["hx"]
        hz=par["hz"]
        D = -2*sparse(I, nx+1, nx+1);
        E = spdiagm(1 => ones(nx),-1=>ones(nx))
        A = (E+D)/hx^2;
        OpOx=A*par["Dox"]; #oxygen diffusion
        OpOx[1,2]=2*par["Dox"]/hx^2;
        OpOx[end,:].=0
        OpOx[end,end]=1
        OpDxx=A # diffusion in space
        OpDxx[1,2]=2/hx^2;
        OpDxx[end,end-1]=2/hx^2;
        D = -2*sparse(I, nz, nz);
        E = spdiagm(1 => ones(nz-1),-1=>ones(nz-1))
        OpDzz=(E+D)/hz^2; # 2nd order differential operator 
        OpDzz[1,1:2]=[-1 1]/hz^2
        OpDzz[end,end-1:end]=[1 -1]/hz^2
        return OpDzz*par["Dz"],OpDxx*par["Dx"],OpOx
end

function formInitialGuess(par,nx,nz)
    K=zeros(nz,nx+1)
    Kvec=zeros((nz+1)*(nx+1)-1)
    n0(x,z)=(tanh((x-0.2)/0.05)+1)*0.8
    uox0(x)=x.^3
    XX=par["spx"]
    indc=-nx
    ZZ=meanval(par["spz"]) 
    for a in 1:length(XX)
        for b in 1:length(ZZ)
            K[b,a]=n0(XX[a],ZZ[b])
        end
        if a<par["nx"]+1
            Kvec[end+indc+a]=uox0(XX[a])
        end
    end
    Kvec[1:end+indc].=vec(K)
    
    return Kvec
end


end