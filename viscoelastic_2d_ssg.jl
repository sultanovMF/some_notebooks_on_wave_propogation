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
α = [1.125 -0.041666666666666664]

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

Npml_y_start = M + 1
Npml_y_end = Ny - M

Nx_start = M + Npml + 1
Nx_end = Nx - M - Npml

Ny_start = M + Npml + 1
Ny_end = Ny - M - Npml


# Perfectly matched layer

κmax = 1

Rc = 0.1 / 100 #процент теоретического отражения
d0 = - 3 * vp * log(Rc) / 2 / Npml

bx = ones(Nx)
ax = ones(Nx)
κx = ones(Nx)

by = ones(Ny)
ay = ones(Ny)
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
    i = M + Npml + 1 - p

    κx[i] = 1 + (κmax - 1) * ((p - 1) / (Npml - 1))^2
    bx[i] = exp(-(d / κ[p] + α) * dt) 
    ax[i] = d / (κ[p] * (d + κ[p] * α)) * (b[p] - 1)
  end

  if PML_TOP_EDGE
    j = Ny - M - Npml + p

    κy[j] = 1 + (κmax - 1) * ((p - 1) / (Npml - 1))^2
    by[j] = exp(-(d / κ[p] + α) * dt) 
    ay[j] = d / (κ[p] * (d + κ[p] * α)) * (b[p] - 1)
  end

  if PML_Bottom_EDGE
    j = M + Npml + 1 - p

    κx[j] = 1 + (κmax - 1) * ((p - 1) / (Npml - 1))^2
    bx[j] = exp(-(d / κ[p] + α) * dt) 
    ax[j] = d / (κ[p] * (d + κ[p] * α)) * (b[p] - 1)
  end
end


ψx_pxx  = zeros(Nx, Ny)
ψx_pxy  = zeros(Nx, Ny)
ψx_vx   = zeros(Nx, Ny)
ψx_vy   = zeros(Nx, Ny)

ψy_pxy  = zeros(Nx, Ny)
ψy_pyy  = zeros(Nx, Ny)
ψy_vy   = zeros(Nx, Ny)
ψy_vx   = zeros(Nx, Ny)

@gif for n in ProgressBar(1:Nt)
  # update particle velocities 
  @tturbo for i in Nx_start:Nx_end
    for j in Ny_start:Ny_end
        pxx_x = 0
        pyy_y = 0
        pxy_x = 0
        pxy_y = 0     

        for m in 1:M
          pxx_x += α[m] * (pxx[i + m, j] - pxx[i- m + 1, j ])
          pyy_y += α[m] * (pyy[i, j + m] - pyy[i , j- m + 1])
          pxy_x += α[m] * (pxy[i + m - 1, j] - pxy[i-m, j])
          pxy_y += α[m] * (pxy[i, j + m - 1] - pxy[i , j- m]) 
        end       
        
        vx[i, j] = vx[i, j] + dt / ρ * (pxx_x / dx + pxy_y / dy)
        vy[i, j] = vy[i, j] + dt / ρ * (pxy_x / dx + pyy_y / dy)
    end 
  end 

  # add source wavelet
  vy[Nx ÷ 2, Ny ÷ 2] = vy[Nx ÷ 2, Ny ÷ 2] + ricker_wavelet(n * dt - 0.1)  

  # update stresses
  @tturbo for i in Nx_start:Nx_end
    for j in Ny_start:Ny_end
        vxx = 0
        vyy = 0        
        vyx = 0
        vxy = 0

        for m in 1:M
          vxx += α[m] * (vx[i+ m - 1, j ] - vx[i-m, j])
          vyy += α[m] * (vy[i, j+ m - 1 ] - vy[i, j-m])  
          vyx += α[m] * (vy[i+m, j] - vy[i- m + 1, j ])
          vxy += α[m] * (vx[i , j+m] - vx[i , j- m + 1])
        end    

        pxx[i, j] = pxx[i, j] + dt  * (  p1P * vxx / dx + p1d * vyy / dy + 0.5 * rxx[i, j])
        pyy[i, j] = pyy[i, j] + dt  * (  p1d * vxx / dx + p1P * vyy / dy + 0.5 * ryy[i, j])
        pxy[i, j] = pxy[i, j] + dt  * (  p1S * vyx / dx + p1S * vxy / dy + 0.5 * rxy[i, j])
    end
  end

  # update memory variables
  @tturbo for i in Nx_start:Nx_end
    for j in Ny_start:Ny_end
        vxx = 0
        vyy = 0        
        vyx = 0
        vxy = 0

        for m in 1:M
          vxx += α[m] * (vx[i + m - 1, j] - vx[i-m, j])
          vyy += α[m] * (vy[i, j + m - 1] - vy[i, j-m])  
          vyx += α[m] * (vy[i+m, j] - vy[i - m + 1, j ])
          vxy += α[m] * (vx[i , j+m] - vx[i , j - m + 1])
        end    
         
        # update stresses
        rxx[i, j] = - 1. / (τ + dt) * (rxx[i, j] + dt  * (  p2P * vxx / dx + p2d * vyy / dy + 0.5 * rxx[i, j]))
        ryy[i, j] = - 1. / (τ + dt) * (ryy[i, j] + dt  * (  p2d * vxx / dx + p2P * vyy / dy + 0.5 * ryy[i, j]))
        rxy[i, j] = - 1. / (τ + dt) * (rxy[i, j] + dt  * (  p2S * vyx / dx + p2S * vxy / dy + 0.5 * rxy[i, j]))

        pxx[i, j] += 0.5 * dt * rxx[i, j]
        pyy[i, j] += 0.5 * dt * ryy[i, j]
        pxy[i, j] += 0.5 * dt * rxy[i, j]
    end
  end

  




  heatmap(vy[:, :],framestyle = :box, clim=(-clip, clip), aspect_ratio = :equal, xlabel = "X", ylabel = "Y", title = "Wave Propagation", color = :seismic, size = (900, 900))
  
  # micro_data[n] = vx[Nx ÷ 2, Ny ÷ 2]
end every 10


#plot(micro)
