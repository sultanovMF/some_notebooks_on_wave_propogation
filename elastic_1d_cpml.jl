using Plots;
using Base.Threads;
using LoopVectorization;
using ProgressBars;

Threads.nthreads() = 8

const vp = 100
# density
const ρ = 10
# lame params
const λ = ρ * vp * vp


const Lx = 100
const Nx = 10000
const dx = Lx / Nx

const T = 5
const Nt = 100000
const dt = T / Nt

@assert vp * dt * sqrt(1.0/dx^2) < 1 "time step is too large, simulation will be unstable", vp * dt * sqrt(1.0/dx^2)

# source
fm = 10
t0 = 0.5
ricker_wavelet(t) = (1. - 2 * pi^2 * fm^2 * t^2) * exp(-pi^2 * fm^2 * t^2)

# main
pxx = zeros(Nx)
vx = zeros(Nx)

# pml
Npml = 20

mem_pxx = zeros(Nx)
mem_vx = zeros(Nx)

κx = ones(Nx)
bx = zeros(Nx)
ax = zeros(Nx)

κmax = 1
αmax = pi * fm / 2
Rc = 0.1 / 100
#d0 = - 3 * vp * log(Rc) / 2 / Npml
d0 = 3000
@show d0
for i in 1:Npml
  poly_i =  (1.0 - (i - 1.0)/ Npml)^2
  pdx = d0 * poly_i
  αx = αmax * ((i - 1) / Npml)^2

  κx[i] = 1.0 + (κmax - 1) * poly_i
  bx[i] = exp(-(pdx / κx[i] + αx) * dt)
  ax[i] = (bx[i] - 1) * pdx / (κx[i] * (pdx + κx[i] * αx))

  κx[Nx - i + 1] = 1.0 + (κmax - 1) * poly_i
  bx[Nx - i + 1] = exp(-(pdx / κx[i] + αx) * dt)
  ax[Nx - i + 1] = (bx[i] - 1) * pdx / (κx[i] * (pdx + κx[i] * αx))
end



xsrc = Nx ÷ 2

x = range(0, Lx, length=Nx)

m = 0

@gif for n in ProgressBar(1:Nt)
  # stresses
  @tturbo for i in 2:Nx
    vx_x = (vx[i] - vx[i - 1]) / dx
    mem_vx[i] = bx[i] * mem_vx[i] + ax[i] * vx_x
    pxx[i] = pxx[i] + dt * λ * (vx_x / κx[i] + mem_vx[i])
  end

  # velocity
  @tturbo for i in 1:Nx-1
    pxx_x = (pxx[i+1] - pxx[i]) / dx

    mem_pxx[i] = (bx[i+1] + bx[i]) / 2 * mem_pxx[i] + (ax[i+1] + ax[i]) / 2  * pxx_x

    vx[i] = vx[i] + dt / ρ * (pxx_x / κx[i] + mem_pxx[i])
  end

  vx[xsrc] += dt / ρ * ricker_wavelet(n * dt - t0)

  vx[1] = 0
  vx[Nx] = 0

  plot(vx, ylim=(-4e-5, 4e-5))

end every 100