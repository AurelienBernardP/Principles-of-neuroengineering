## Model gating functions
# All activation and inactivation curves are defined by the Boltzman function
Xinf(V,A,B)=1/(1+exp((V+A)/B))
function invXinf(m,A,B)
    if m <= 0
        B*log(1/0.0001 - 1) - A
    elseif m >= 1
        B*log(1/0.9999 - 1) - A
    else
        B*log(1/m - 1) - A
    end
end

# All timeconstant curves are defined by the shifted Boltzman function
tauX(V,A,B,D,E)=A-B/(1+exp((V+D)/E))

# Sodium current
mNainf(V) = Xinf(V,25.,-5.); tau_mNa(V) = tauX(V,0.75,0.5,100.,-20.)
hNainf(V) = Xinf(V,40.,10.); tau_hNa(V) = tauX(V,4.0,3.5,50.,-20.)

# Potassium currents
mKdinf(V) = Xinf(V,15.,-10.); tau_mKd(V) = tauX(V,5.0,4.5,30.,-20.)

mAfinf(V) = Xinf(V,80,-10.); tau_mAf(V) = tauX(V,0.75,0.5,100.,-20.)
hAfinf(V) = Xinf(V,60,5.); tau_hAf(V) = 10*tauX(V,0.75,0.5,100.,-20.)

mAsinf(V) = Xinf(V,60,-10.); tau_mAs(V) = 10*tauX(V,0.75,0.5,100.,-20.)
hAsinf(V) = Xinf(V,20,5.); tau_hAs(V) = 100*tauX(V,0.75,0.5,100.,-20.)

mKCainf(Ca) = Xinf(Ca,-30.0,-10.); tau_mKCa = 500.

# Calcium currents
mCaLinf(V) = Xinf(V,45.,-5.); tau_mCaL(V) = tauX(V,6.0,5.5,30.,-20.)

# mCaLinf(V) = Xinf(V,35.,-5.); for Type V 
mCaTinf(V) = Xinf(V,55,-5.); tau_mCaT(V) = tauX(V,6.0,5.5,30.,-20.)
hCaTinf(V) = Xinf(V,70,10.); tau_hCaT(V) = 100*tauX(V,6.0,5.5,30.,-20.)

# Cation current (H-current)
mHinf(V) = Xinf(V,85,10.); tau_mH(V) = 50*tauX(V,6.0,5.5,30.,-20.);

# Bursts-of-bursts-of-bursts currents (proof-of-concept)
mKinf(V) = Xinf(V,15.,-5.); tau_mK(V) = tauX(V,5.0,4.5,30.,-20.)
mKsinf(V) = Xinf(V,35.,-5.); tau_mKs(V) = 10*tauX(V,5.0,4.5,30.,-20.)
mKssinf(V) = Xinf(V,40.3,-5.); tau_mKss(V) = 400*tauX(V,5.0,4.5,30.,-20.)
mKsssinf(V) = Xinf(V,50.3,-5.); tau_mKsss(V) = 40*400*tauX(V,5.0,4.5,30.,-20.)
mCainf(V) = Xinf(V,35.,-5.); tau_mCa(V) = tauX(V,6.0,5.5,30.,-20.)
mCasinf(V) = Xinf(V,40,-5.); tau_mCas(V) = 20*tauX(V,6.0,5.5,30.,-20.)
mCassinf(V) = Xinf(V,50,-5.); tau_mCass(V) = 20*20*tauX(V,6.0,5.5,30.,-20.)

## Equivalent potentials
Vf(m) = invXinf(m,25.,-5.) # fast variable = mNa
Vs(m) = invXinf(m,15.,-10.) # slow variable = mKd
Vu(m) = invXinf(m,70.,10.) # ultraslow variable = hCaT

## Model parameters
const VNa = 40.; # Sodium reversal potential
const VK = -90.; # Potassium reversal potential
const VCa = 120.; # Calcium reversal potential
const VH= -40.; # Reversal potential for the H-current (permeable to both sodium and potassium ions)
const Vl = -50.; # Reversal potential of leak channels

const C=1.; # Membrane capacitance
const α=0.1; # Calcium dynamics (L-current)
const β=0.1 # Calcium dynamics (T-current)
const tau_Ca = 500. # Time-constant of calcium dynamics


## Simulation function in current-clamp mode
function neuron_CC_red(du,u,p,t)
    # Stimulations parameters
    Iapp=p[1] # Amplitude of constant applied current
    It=p[2] # Time-varying stimulation current

    # Maximal conductances
    gNa=p[3] # Sodium current maximal conductance
    gKd=p[4]  # Delayed-rectifier potassium current maximal conductance
    gAf=p[5] # Fast A-type potassium current maximal conductance
    gAs=p[6] # Slow A-type potassium current maximal conductance
    gKCa=p[7] # Calcium-activated potassium current maximal conductance
    gCaL=p[8] # L-type calcium current maximal conductance
    gCaT=p[9] # T-type calcium current maximal conductance
    gH=p[10] # H-current maximal conductance
    gl=p[11] # Leak current maximal conductance

    # Variables
    V=u[1] # Membrane potential
    mKd=u[2] # Delayed-rectifier potassium current activation
    hCaT=u[3] # T-type calcium current inactivation
    Ca=u[4] # Intracellular calcium concentration

    # ODEs
                    # Sodium current
    du[1] = (1/C) * (-gNa*mNainf(V)*hNainf(Vs(mKd))*(V-VNa) +
                    # Potassium Currents
                    -gKd*mKd*(V-VK) -gAf*mAfinf(V)*hAfinf(Vs(mKd))*(V-VK) -gAs*mAsinf(Vs(mKd))*hAsinf(Vu(hCaT))*(V-VK) +
                    -gKCa*mKCainf(Ca)*(V-VK) +
                    # Calcium currents
                    -gCaL*mCaLinf(Vs(mKd))*(V-VCa) +
                    -gCaT*mCaTinf(Vs(mKd))*hCaT*(V-VCa) +
                    # Cation current
                    -gH*mHinf(Vu(hCaT))*(V-VH) +
                    # Passive currents
                    -gl*(V-Vl) +
                    # Stimulation currents
                    +Iapp + It(t))
    du[2] = (1/tau_mKd(V)) * (mKdinf(V) - mKd)
    du[3] = (1/tau_hCaT(V)) * (hCaTinf(V) - hCaT)
    du[4] = (1/tau_Ca) * ((-α*gCaL*mCaLinf(Vs(mKd))*(V-VCa))+(-β*gCaT*mCaTinf(Vs(mKd))*hCaT*(V-VCa)) - Ca)
end

const VNoise=0.01
const CaNoise=0.01

function simple_lag(du,u,p,t)
    τ = 1/p[1]
    du[1] = -1/τ*u[1]
end

function simple_lag_noise(du,u,p,t)
    τ = 1/p[1]
    du[1] = 1/τ
end


## Simulation function in voltage-clamp mode
function neuron_VC_simu(du,u,p,t)
    # Stimulations parameters
    V=p[1] # Command voltage
    
    # Maximal conductances
    gNa=p[2] # Sodium current maximal conductance
    gKd=p[3]  # Delayed-rectifier potassium current maximal conductance
    gAf=p[4] # Fast A-type potassium current maximal conductance
    gAs=p[5] # Slow A-type potassium current maximal conductance
    gKCa=p[6] # Calcium-activated potassium current maximal conductance
    gCaL=p[7] # L-type calcium current maximal conductance
    gCaT=p[8] # T-type calcium current maximal conductance
    gH=p[9] # H-current maximal conductance
    gl=p[10] # Leak current maximal conductance

    # Variables
    mNa=u[1] # Sodium current activation
    hNa=u[2] # Sodium current inactivation
    mKd=u[3] # Delayed-rectifier potassium current activation
    mAf=u[4] # Fast A-type potassium current activation
    hAf=u[5] # Fast A-type potassium current inactivation
    mAs=u[6] # Slow A-type potassium current activation
    hAs=u[7] # Slow A-type potassium current inactivation
    mCaL=u[8] # L-type calcium current activation
    mCaT=u[9] # T-type calcium current activation
    hCaT=u[10] # T-type calcium current inactivation
    mH=u[11] # H current activation
    Ca=u[12] # Intracellular calcium concentration

    # ODEs
    du[1] = (1/tau_mNa(V(t))) * (mNainf(V(t)) - mNa)
    du[2] = (1/tau_hNa(V(t))) * (hNainf(V(t)) - hNa)
    du[3] = (1/tau_mKd(V(t))) * (mKdinf(V(t)) - mKd)
    du[4] = (1/tau_mAf(V(t))) * (mAfinf(V(t)) - mAf)
    du[5] = (1/tau_hAf(V(t))) * (hAfinf(V(t)) - hAf)
    du[6] = (1/tau_mAs(V(t))) * (mAsinf(V(t)) - mAs)
    du[7] = (1/tau_hAs(V(t))) * (hAsinf(V(t)) - hAs)
    du[8] = (1/tau_mCaL(V(t))) * (mCaLinf(V(t)) - mCaL)
    du[9] = (1/tau_mCaT(V(t))) * (mCaTinf(V(t)) - mCaT)
    du[10] = (1/tau_hCaT(V(t))) * (hCaTinf(V(t)) - hCaT)
    du[11] = (1/tau_mH(V(t))) * (mHinf(V(t)) - mH)
    du[12] = (1/tau_Ca) * ((-α*gCaL*mCaL*(V(t)-VCa))+(-β*gCaT*mCaT*hCaT*(V(t)-VCa)) - Ca)
end

function neuron_VC(p)
    # Stimulations parameters
    V=p[1] # Command voltage
    
    # Maximal conductances
    gNa=p[2] # Sodium current maximal conductance
    gKd=p[3]  # Delayed-rectifier potassium current maximal conductance
    gAf=p[4] # Fast A-type potassium current maximal conductance
    gAs=p[5] # Slow A-type potassium current maximal conductance
    gKCa=p[6] # Calcium-activated potassium current maximal conductance
    gCaL=p[7] # L-type calcium current maximal conductance
    gCaT=p[8] # T-type calcium current maximal conductance
    gH=p[9] # H-current maximal conductance
    gl=p[10] # Leak current maximal conductance
    
    # Simulation time
    Tfinal=p[11]
    tspan=(0.0,Tfinal)
    
    # Initial conditions
    V0=V(0); mNa0=mNainf(V0); hNa0=hNainf(V0); mKd0=mKdinf(V0); mAf0=mAfinf(V0); hAf0=hAfinf(V0);
    mAs0=mAsinf(V0); hAs0=hAsinf(V0); mCaL0=mCaLinf(V0); mCaT0=mCaTinf(V0); hCaT0=hCaTinf(V0); mH0=mHinf(V0) 
    Ca0=(-α*gCaL*mCaL0*(V0-VCa))+(-β*gCaT*mCaT0*hCaT0*(V0-VCa))

    x0 = [mNa0; hNa0; mKd0; mAf0; hAf0; mAs0; hAs0; mCaL0; mCaT0; hCaT0; mH0; Ca0]
    
    # Parameter vector for simulations
    p=(V,gNa,gKd,gAf,gAs,gKCa,gCaL,gCaT,gH,gl)

    # Simulation
    prob = ODEProblem(neuron_VC_simu,x0,tspan,p) # Simulation of ODEs
    sol = solve(prob,Euler(),dt=1e-3);
    
    t=sol.t
    Vt=V.(t)
    mNa=sol[1,:] # Sodium current activation
    hNa=sol[2,:] # Sodium current inactivation
    mKd=sol[3,:] # Delayed-rectifier potassium current activation
    mAf=sol[4,:] # Fast A-type potassium current activation
    hAf=sol[5,:] # Fast A-type potassium current inactivation
    mAs=sol[6,:] # Slow A-type potassium current activation
    hAs=sol[7,:] # Slow A-type potassium current inactivation
    mCaL=sol[8,:] # L-type calcium current activation
    mCaT=sol[9,:] # T-type calcium current activation
    hCaT=sol[10,:] # T-type calcium current inactivation
    mH=sol[11,:] # H current activation
    Ca=sol[12,:] # Intracellular calcium concentration
    
    Iclamp=zeros(length(sol.t))
    
                    # Sodium current
    @. Iclamp = (-1/C) * (-gNa*mNa*hNa*(Vt-VNa) +
                    # Potassium Currents
                    -gKd*mKd*(Vt-VK) -gAf*mAf*hAf*(Vt-VK) -gAs*mAs*hAs*(Vt-VK) +
                    -gKCa*mKCainf(Ca)*(Vt-VK) +
                    # Calcium currents
                    -gCaL*mCaL*(Vt-VCa) +
                    -gCaT*mCaT*hCaT*(Vt-VCa) +
                    # Cation current
                    -gH*mH*(Vt-VH) +
                    # Passive currents
                    -gl*(Vt-Vl))
    
    return (t, Vt, Iclamp)
end

function neuron_IV(p)
    
    # Stimulations parameters
    V0=p[1] # Command voltage
    
    # Maximal conductances
    gNa=p[2] # Sodium current maximal conductance
    gKd=p[3]  # Delayed-rectifier potassium current maximal conductance
    gAf=p[4] # Fast A-type potassium current maximal conductance
    gAs=p[5] # Slow A-type potassium current maximal conductance
    gKCa=p[6] # Calcium-activated potassium current maximal conductance
    gCaL=p[7] # L-type calcium current maximal conductance
    gCaT=p[8] # T-type calcium current maximal conductance
    gH=p[9] # H-current maximal conductance
    gl=p[10] # Leak current maximal conductance
    
    # Simulation time
    Tfinal=p[11]
    tspan=(0.0,Tfinal)
    
    # Time snapshots
    t_snap=p[12]
    n_snap=length(t_snap)
    
    IVs=zeros(n_snap,Vvec_IV_cbm_length)
    gVs=zeros(n_snap,Vvec_IV_cbm_length-1)
    
    for i=1:Vvec_IV_cbm_length
        Vcmd(t) = V0*(t<=1.0) + Vvec_IV_cbm[i]*(t>1.0) 
        q = (Vcmd,gNa,gKd,gAf,gAs,gKCa,gCaL,gCaT,gH,gl,Tfinal)
        (t, Vt, Iclamp) = neuron_VC(q)
        #println(t)
        for j=1:n_snap
            tj=findfirst(x -> x>t_snap[j]+1.0, t)
            IVs[j,i]=Iclamp[tj]
        end
    end
    
    for i=1:Vvec_IV_cbm_length-1
        for j=1:n_snap
            gVs[j,i]=(IVs[j,i+1]-IVs[j,i])/(Vvec_IV_cbm[2]-Vvec_IV_cbm[1])
        end
    end
    
    return (IVs,gVs)
    
end

