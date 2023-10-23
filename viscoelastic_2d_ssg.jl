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
T  = 0.55 # sec
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

σxx_x = 0
σxy_y = 0
σxz_z = 0

fm = 40

#ricker_wavelet(t) = 2. / (sqrt(3) * sigma * pi^0.25) * (1 - (t/sigma)^2) * exp(-t^2 / 2 / sigma^2)
ricker_wavelet(t) = (1. - 2 * pi^2 * fm^2 * t^2) * exp(-pi^2 * fm^2 * t^2)

xs = range(0, stop=Lx, length=Nx)
ys = range(0, stop=Ly, length=Ny)

micro = zeros(Nt)
# vx[Nx ÷ 2,  Ny ÷ 2] = 200

@gif for n in ProgressBar(1:Nt)
  # update particle velocities 
  @tturbo for i in 2:Ny-2
    for j in 5:Nx-5   
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
  @tturbo for i in 5:Ny-5
    for j in 5:Nx-5
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
  @tturbo for i in 5:Ny-5
    for j in 5:Nx-5
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
  micro[n] = vx[Nx ÷ 2, Ny ÷ 2]
end every 10


#plot(micro)
