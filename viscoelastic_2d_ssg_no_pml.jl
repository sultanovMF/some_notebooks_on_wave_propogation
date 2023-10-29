using Plots;
using Base.Threads;
using LoopVectorization;
using ProgressBars;

Threads.nthreads() = 8

# viscoelastic params
τP =0
τS = 0
τ =1
# τP = 0.353
# τS = 0.352
# τ = 0.351

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

@assert vp * dt * sqrt(1/dx^2 + 1/dy^2) < 1 "no stability"
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

fm = 40# for ricker wavelet

ricker_wavelet(t) = (1. - 2 * pi^2 * fm^2 * t^2) * exp(-pi^2 * fm^2 * t^2)

micro_coord = [Nx ÷ 2, Ny ÷ 2]
micro_data = zeros(Nt)

# PML настрйока
Npml = 5 # толщина PML слоя

Nx_start = M  + 1
Nx_end = Nx - M

Ny_start = M + 1
Ny_end = Ny - M


@gif for n in ProgressBar(1:Nt)
  # update stresses
  @tturbo for i in 2:Nx
    for j in 2:Ny
        vxx = 0
        vyy = 0        

        vxx += (vx[i, j] - vx[i - 1, j]) / dx
        vyy += (vy[i, j] - vy[i, j - 1]) / dy 

        pxx[i, j] = pxx[i, j] + dt  * (p1P * (vxx ) + p1d * (vyy) + 0.5 * rxx[i, j])
        pyy[i, j] = pyy[i, j] + dt  * (p1d * (vxx) + p1P * (vyy) + 0.5 * ryy[i, j])


        rxx[i, j] = - 1. / (τ + 0.5 * dt) * (τ * rxx[i, j] + dt  * (p2P * vxx + p2d * vyy + 0.5 * rxx[i, j]))
        ryy[i, j] = - 1. / (τ + 0.5 * dt) * (τ * ryy[i, j] + dt  * (p2d * vxx + p2P * vyy + 0.5 * ryy[i, j]))


        pxx[i, j] += 0.5 * dt * rxx[i, j]
        pyy[i, j] += 0.5 * dt * ryy[i, j]
    end
  end

  @tturbo for i in 1:Nx-1
    for j in 1:Ny-1 
        vyx = (vy[i + 1, j] - vy[i, j]) /dx
        vxy = (vx[i, j + 1] - vx[i, j]) /dy

        pxy[i, j] = pxy[i, j] + dt  * (p1S * (vyx ) + p1S * (vxy) + 0.5 * rxy[i, j])

        rxy[i, j] = - 1. / (τ + 0.5 * dt) * (τ * rxy[i, j] + dt  * (p2S * vyx + p2S * vxy + 0.5 * rxy[i, j]))

        pxy[i, j] += 0.5 * dt * rxy[i, j]
    end
  end

  # update particle velocities 
  @tturbo for i in 1:Nx-1
    for j in 2:Ny-1
        pxx_x = (pxx[i + 1, j] - pxx[i, j ]) / dx
        pxy_y = (pxy[i, j] - pxy[i , j - 1]) /dy

        vx[i, j] = vx[i, j] + dt / ρ * (pxx_x + pxy_y)

    end 
  end 

  @tturbo for i in 2:Nx
    for j in 1:Ny-1
        pyy_y = (pyy[i, j + 1] - pyy[i , j]) / dy
        pxy_x = (pxy[i, j] - pxy[i - 1, j]) / dx
    
        vy[i, j] = vy[i, j] + dt / ρ * (pxy_x + pyy_y)

    end 
  end 

  vx[1, :] .= 0
  vx[Nx, :] .= 0
  vx[:, 1] .= 0
  vx[:, Ny] .= 0

  vy[1, :] .= 0
  vy[Nx, :] .= 0
  vy[:, 1] .= 0
  vy[:, Ny] .= 0

  # add source wavelet
  vy[Nx ÷ 2, Ny ÷ 2] = vy[Nx ÷ 2, Ny ÷ 2] + dt / ρ * ricker_wavelet(n * dt - 0.1)  

  heatmap(vy[:, :],framestyle = :box, aspect_ratio = :equal, xlabel = "X", ylabel = "Y", title = "Wave Propagation", color = :seismic, size = (900, 900))
  
end every 10


