# Packages 
using DifferentialEquations, Plots, Plots.PlotMeasures, LaTeXStrings, SymPy, LsqFit, Interpolations, Roots, ColorSchemes, LinearAlgebra, DelimitedFiles

# Functions
heaviside(t)=0*(t<0)+1*(t>=0)
boltz(x,A,B)=1.0/( 1.0 + exp(-B*(x-A)/4.0) )
dboltzdV(x,A,B)=(B*exp((B*(A - x))/4.0))/(4.0*(exp((B*(A - x))/4.0) + 1.0)^2.0)
boltz_inv(x,A,B)=4*log(1/x-1)/B+A
pulse(t,ti,tf)=heaviside(t-ti)-heaviside(t-tf)
Xinf(V,A,B)=1/(1+exp((V+A)/B))
function Xinf_inv(V,A,B)
    if imag(log(Complex(1/V-1))) == 0
        return B*log(1/V-1)-A
    else
        return NaN
    end
end