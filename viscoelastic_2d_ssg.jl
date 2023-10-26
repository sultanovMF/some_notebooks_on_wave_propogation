using Plots;
using Base.Threads;
using LoopVectorization;
using ProgressBars;

Threads.nthreads() = 8

# wavefield clip
clip = 2.5e-2

# viscoelastic params
τP = 0
τS = 0
τ = 10

#initial velocities
vp = 2955.0
vs = 2362.0
# density
ρ = 7100.0
# lame params
μ = ρ * vs * vs 
λ = ρ * vp * vp - 2 * μ


# Настройки сетки
Lx = 3005 # m
Ly = 3005 # m

Nx = 601 # points
Ny = 601 # points

dx = Lx / Nx # m
dy = Ly / Ny # m

# Настройки временной сетки
T  = 1 # sec
dt = 0.6e-3
Nt = trunc(Int, T / dt)
# Nt = 1000
# dt = T / Nt

# source wavelet params
fc = 17.0
tshift = 0.0
ts = 1.0 / fc

# M = 6 # order of approx
# α = [1.221336364746061, -0.0969314575195135, 0.017447662353510373, -0.0029672895159030804, 0.00035900539822037244, -2.1847811612209015e-5] # spacial finite difference coefficient

M = 2
fd_coef = [1.125 -0.041666666666666664]

vx = zeros(Nx, Ny)
vy = zeros(Nx, Ny)

pxx = zeros(Nx, Ny)
pyy = zeros(Nx, Ny)
pxy = zeros(Nx, Ny)

rxx = zeros(Nx, Ny)
ryy = zeros(Nx, Ny)
rxy = zeros(Nx, Ny)

p1P = (λ + 2 * μ) * (1 + τP)
p1S = μ * (1 + τS)
p1d = p1P - 2 * p1S

p2P = (λ + 2 * μ) * τP
p2S = μ * τS
p2d = p2P - 2 * p2S

fm = 40 # for ricker wavelet

ricker_wavelet(t) = (1. - 2 * pi^2 * fm^2 * t^2) * exp(-pi^2 * fm^2 * t^2)

micro_coord = [Nx ÷ 2, Ny ÷ 2]
micro_data = zeros(Nt)

# PML настрйока
Npml = 10 # толщина PML слоя

Nx_start = M  + 1
Nx_end = Nx - M

Ny_start = M + 1
Ny_end = Ny - M


# Perfectly matched layer

κmax = 1

Rc = 0.1 / 100 #процент теоретического отражения
d0 = - 3 * vp * log(Rc) / 2 / Npml

bx = zeros(Nx)
ax = zeros(Nx)
κx = ones(Nx)

by = zeros(Ny)
ay = zeros(Ny)
κy = ones(Ny)

αmax = pi * 25

PML_RIGHT_EDGE = true
PML_LEFT_EDGE = true
PML_TOP_EDGE = true
PML_BOTTOM_EDGE = true


for p in 1:Npml
  d = d0 * ((p - 1) / (Npml - 1))^2
  α = αmax - αmax * (p - 1) / (Npml - 1)

  if PML_RIGHT_EDGE
    i = Nx - M - Npml + p

    κx[i] = 1 + (κmax - 1) * ((p - 1) / (Npml - 1))^2
    bx[i] = exp(-(d / κx[p] + α) * dt) 
    ax[i] = d / (κx[p] * (d + κx[p] * α)) * (bx[p] - 1)
  end

  if PML_LEFT_EDGE
    local i = M + Npml + 1 - p
    
    κx[i] = 1 + (κmax - 1) * ((p - 1) / (Npml - 1))^2
    bx[i] = exp(-(d / κx[p] + α) * dt) 
    ax[i] = d / (κx[p] * (d + κx[p] * α)) * (bx[p] - 1)
  end

  if PML_TOP_EDGE
    local j = Ny - M - Npml + p

    κy[j] = 1 + (κmax - 1) * ((p - 1) / (Npml - 1))^2
    by[j] = exp(-(d / κy[p] + α) * dt) 
    ay[j] = d / (κy[p] * (d + κy[p] * α)) * (by[p] - 1)
  end

  if PML_BOTTOM_EDGE
    local j = M + Npml + 1 - p

    κy[j] = 1 + (κmax - 1) * ((p - 1) / (Npml - 1))^2
    by[j] = exp(-(d / κy[p] + α) * dt) 
    ay[j] = d / (κy[p] * (d + κy[p] * α)) * (by[p] - 1)
  end
end

# для интерполяции в промежуточных значениях
if PML_RIGHT_EDGE
  local i = Nx - M + 1

  κx[i] = κx[i - 1]
  bx[i] = bx[i - 1]
  ax[i] = ax[i - 1]
end

if PML_LEFT_EDGE
  local i = M + Npml + 1

  κx[i] = κx[i - 1]
  bx[i] = bx[i - 1]
  ax[i] = ax[i - 1]
end

if PML_TOP_EDGE
  local j = Ny - M + 1

  κy[j] = κy[j - 1]
  by[j] = by[j - 1]
  ay[j] = ay[j - 1]
end

if PML_BOTTOM_EDGE
  local j = M + Npml + 1

  κy[j] = κy[j - 1]
  by[j] = by[j - 1]
  ay[j] = ay[j - 1]
end

ψx_pxx   = zeros(Nx, Ny)
ψx_pxy   = zeros(Nx, Ny)
ψx_vxx   = zeros(Nx, Ny)
ψx_vyx   = zeros(Nx, Ny)

ψy_pxy   = zeros(Nx, Ny)
ψy_pyy   = zeros(Nx, Ny)
ψy_vyy   = zeros(Nx, Ny)
ψy_vxy   = zeros(Nx, Ny)

@gif for n in ProgressBar(1:Nt)
  # update stresses
  @tturbo for i in Nx_start:Nx_end
    for j in Ny_start:Ny_end
        vxx = 0
        vyy = 0        
        vyx = 0
        vxy = 0

        for m in 1:M
          vxx += fd_coef[m] * (vx[i + m - 1, j ] - vx[i-m, j]) / dx
          vyy += fd_coef[m] * (vy[i, j + m - 1 ] - vy[i, j-m]) / dy
          vyx += fd_coef[m] * (vy[i+m, j] - vy[i- m + 1, j ]) /dx
          vxy += fd_coef[m] * (vx[i , j+m] - vx[i , j- m + 1]) /dy
        end    

        ψx_vxx[i, j] = bx[i] * ψx_vxx[i, j] + ax[i] * vxx
        ψy_vyy[i, j] = by[j] * ψy_vyy[i, j] + ay[j] * vyy

        ψx_vyx[i, j] = (bx[i] + bx[i+1]) / 2 * ψx_vyx[i, j] + (ax[i] + ax[i + 1]) / 2 * vyx
        ψy_vxy[i, j] = (by[j] + by[j+1]) / 2 * ψy_vxy[i, j] + (ay[j] + ay[j + 1]) / 2 * vxy

        pxx[i, j] = pxx[i, j] + dt  * (p1P * (vxx / κx[i] +  ψx_vxx[i, j]) + p1d * (vyy / κy[j] + ψy_vyy[i, j]) + 0.5 * rxx[i, j])
        pyy[i, j] = pyy[i, j] + dt  * (p1d * (vxx / κx[i] +  ψx_vxx[i, j]) + p1P * (vyy / κy[j] + ψy_vyy[i, j]) + 0.5 * ryy[i, j])
        pxy[i, j] = pxy[i, j] + dt  * (p1S * (vyx / κx[i] +  ψx_vyx[i, j]) + p1S * (vxy / κy[j] + ψy_vxy[i, j]) + 0.5 * rxy[i, j])

        rxx[i, j] = - 1. / (τ + dt) * (rxx[i, j] + dt  * (p2P * vxx + p2d * vyy + 0.5 * rxx[i, j]))
        ryy[i, j] = - 1. / (τ + dt) * (ryy[i, j] + dt  * (p2d * vxx + p2P * vyy + 0.5 * ryy[i, j]))
        rxy[i, j] = - 1. / (τ + dt) * (rxy[i, j] + dt  * (p2S * vyx + p2S * vxy + 0.5 * rxy[i, j]))

        pxx[i, j] += 0.5 * dt * rxx[i, j]
        pyy[i, j] += 0.5 * dt * ryy[i, j]
        pxy[i, j] += 0.5 * dt * rxy[i, j]
    end
  end

  # update particle velocities 
  @tturbo for i in Nx_start:Nx_end
    for j in Ny_start:Ny_end
        pxx_x = 0
        pyy_y = 0
        pxy_x = 0
        pxy_y = 0     

        for m in 1:M
          pxx_x += fd_coef[m] * (pxx[i + m, j] - pxx[i- m + 1, j ]) / dx
          pyy_y += fd_coef[m] * (pyy[i, j + m] - pyy[i , j- m + 1]) / dy
          pxy_x += fd_coef[m] * (pxy[i + m - 1, j] - pxy[i-m, j]) / dx
          pxy_y += fd_coef[m] * (pxy[i, j + m - 1] - pxy[i , j- m]) /dy
        end       

        ψx_pxx[i, j] = (bx[i] + bx[i+1]) / 2 * ψx_pxx[i, j] + (ax[i] + ax[i]) / 2 * pxx_x
        ψy_pxy[i, j] = by[j] * ψy_pxy[i, j] + ay[j] * pxy_y
        vx[i, j] = vx[i, j] + dt / ρ * (pxx_x / κx[i] + ψx_pxx[i, j] + pxy_y / κy[j] + ψy_pxy[i, j])

        ψx_pxy[i, j] = bx[i] * ψx_pxy[i, j] + ax[i] * pxy_x
        ψy_pyy[i, j] = (by[j] + by[j + 1]) / 2 * ψy_pyy[i, j] + (ay[j] + ay[j+1]) / 2 * pyy_y
        vy[i, j] = vy[i, j] + dt / ρ * (pxy_x / κx[i] + ψx_pxy[i, j] + pyy_y / κy[j] + ψy_pyy[i, j])
    end 
  end 

  # add source wavelet
  vy[Nx ÷ 2, Ny ÷ 2] = vy[Nx ÷ 2, Ny ÷ 2] + ricker_wavelet(n * dt - 0.1)  

  heatmap(vy[:, :],framestyle = :box, clim=(-clip, clip), aspect_ratio = :equal, xlabel = "X", ylabel = "Y", title = "Wave Propagation", color = :seismic, size = (900, 900))
  
  # micro_data[n] = vx[Nx ÷ 2, Ny ÷ 2]
end every 10


#plot(micro)
