using Plots;
using Base.Threads;
using LoopVectorization;
using ProgressBars;



vp = 2955.0
vs = 2362.0
# density
ρ = 7100.0
# lame params
μ = ρ * vs * vs 
λ = ρ * vp * vp - 2 * μ


Lx = 3000