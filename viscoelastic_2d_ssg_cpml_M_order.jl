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

M = 7 # order of approx
fd_coef = [1.228606224061052, -0.1023838520056638, 0.02047677040126517, -0.004178932734957832, 0.0006894535488655083, -7.692250338584502e-5, 4.236514751792867e-6] # spacial finite difference coefficient

# M = 2
# fd_coef = [1.125 -0.041666666666666664]

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
Npml = 30 # толщина PML слоя

ψ_vx_x = zeros(Nx, Ny)
ψ_vy_x = zeros(Nx, Ny)
ψ_vx_y = zeros(Nx, Ny)
ψ_vy_y = zeros(Nx, Ny)

ψ_pxx_x = zeros(Nx, Ny)
ψ_pxy_x = zeros(Nx, Ny)
ψ_pxy_y = zeros(Nx, Ny)
ψ_pyy_y = zeros(Nx, Ny)

κx = ones(Nx)
bx = zeros(Nx)
ax = zeros(Nx)

κy = ones(Ny)
by = zeros(Ny)
ay = zeros(Ny)

κmax = 20
αmax = pi * fm
Rc = 0.0001 / 100
#d0 = - 3 * vp * log10(Rc) / 2 / Npml

d0 = 3000
@show d0
for i in 1:Npml
  pdx = d0 * (1.0 - (i - 1.0)/ Npml)^2
  αx = αmax * ((i - 1) / Npml)^2

  κx[i] = 1.0 + (κmax - 1) * (1.0 - (i - 1.0)/ Npml)^2
  bx[i] = exp(-(pdx / κx[i] + αx) * dt)
  ax[i] = (bx[i] - 1) * pdx / (κx[i] * (pdx + κx[i] * αx))

  κx[Nx - i + 1] = κx[i]
  bx[Nx - i + 1] = bx[i]
  ax[Nx - i + 1] = ax[i]

  κy[i] = κx[i]
  by[i] = bx[i]
  ay[i] = ax[i]

  κy[Ny - i + 1] = κx[i]
  by[Ny - i + 1] = bx[i]
  ay[Ny - i + 1] = ax[i]

end

@gif for n in ProgressBar(1:Nt)
  # update stresses
  @tturbo for i in 1+M:Nx-M+1
    for j in 1+M:Ny-M+1
        vxx = 0
        vyy = 0        
        
        for m in 1:M
          vxx += fd_coef[m] * (vx[i + m - 1, j ] - vx[i - m, j]) / dx
          vyy += fd_coef[m] * (vy[i, j + m - 1 ] - vy[i, j - m]) / dy
        end    

        ψ_vx_x[i, j] = bx[i] * ψ_vx_x[i, j] + ax[i] * vxx
        ψ_vy_y[i, j] = by[j] * ψ_vy_y[i, j] + ay[j] * vyy

        pxx[i, j] = pxx[i, j] + dt  * (p1P * (vxx / κx[i] + ψ_vx_x[i, j]) + p1d * (vyy / κy[j] + ψ_vy_y[i, j]) + 0.5 * rxx[i, j])
        pyy[i, j] = pyy[i, j] + dt  * (p1d * (vxx / κx[i] + ψ_vx_x[i, j]) + p1P * (vyy / κy[j] + ψ_vy_y[i, j]) + 0.5 * ryy[i, j])


        rxx[i, j] = - 1. / (τ + 0.5 * dt) * (τ * rxx[i, j] + dt  * (p2P * (vxx / κx[i] + ψ_vx_x[i, j]) + p2d *  (vyy / κy[j] + ψ_vy_y[i, j]) + 0.5 * rxx[i, j]))
        ryy[i, j] = - 1. / (τ + 0.5 * dt) * (τ * ryy[i, j] + dt  * (p2d * (vxx / κx[i] + ψ_vx_x[i, j]) + p2P *  (vyy / κy[j] + ψ_vy_y[i, j]) + 0.5 * ryy[i, j]))


        pxx[i, j] += 0.5 * dt * rxx[i, j]
        pyy[i, j] += 0.5 * dt * ryy[i, j]
    end
  end

  @tturbo for i in M:Nx-M
    for j in M:Ny-M
        vyx = 0
        vxy = 0

        for m in 1:M
          vyx += fd_coef[m] * (vy[i + m, j] - vy[i - m + 1, j ]) /dx
          vxy += fd_coef[m] * (vx[i , j + m] - vx[i , j- m + 1]) /dy
        end    

        ψ_vy_x[i, j] = (bx[i+1] + bx[i]) / 2 * ψ_vy_x[i, j] + (ax[i+1] + ax[i]) / 2 * vyx
        ψ_vx_y[i, j] = (by[j+1] + by[j]) / 2 * ψ_vx_y[i, j] + (ay[j+1] + ay[j]) / 2 * vxy

        pxy[i, j] = pxy[i, j] + dt  * (p1S * (vyx / κx[i] + ψ_vy_x[i, j]) + p1S * (vxy / κy[j] + ψ_vx_y[i, j]) + 0.5 * rxy[i, j])

        rxy[i, j] = - 1. / (τ + 0.5 * dt) * (τ * rxy[i, j] + dt  * (p2S * (vyx / κx[i] + ψ_vy_x[i, j]) + p2S * (vxy / κy[j] + ψ_vx_y[i, j]) + 0.5 * rxy[i, j]))

        pxy[i, j] += 0.5 * dt * rxy[i, j]
    end
  end

  # update particle velocities 
  @tturbo for i in M:Nx-M
    for j in 1+M:Ny-M+1
      pxx_x = 0
      pxy_y = 0     

      for m in 1:M
        pxx_x += fd_coef[m] * (pxx[i + m, j] - pxx[i - m + 1, j ]) / dx
        pxy_y += fd_coef[m] * (pxy[i, j + m - 1] - pxy[i , j - m]) / dy
      end       

      ψ_pxx_x[i, j] = (bx[i+1] + bx[i]) / 2 * ψ_pxx_x[i, j] + (ax[i+1] + ax[i]) / 2 * pxx_x
      ψ_pxy_y[i, j] = by[j] * ψ_pxy_y[i, j] + ay[j] * pxy_y

      vx[i, j] = vx[i, j] + dt / ρ * (pxx_x / κx[i] + ψ_pxx_x[i, j] + pxy_y / κy[j] + ψ_pxy_y[i, j])

    end 
  end 

  @tturbo for i in 1+M:Nx-M+1
    for j in M:Ny-M
        pyy_y = 0
        pxy_x = 0

        for m in 1:M
          pyy_y += fd_coef[m] * (pyy[i, j + m] - pyy[i , j- m + 1]) / dy
          pxy_x += fd_coef[m] * (pxy[i + m - 1, j] - pxy[i-m, j]) / dx
        end       

        ψ_pxy_x[i, j] = bx[i] * ψ_pxy_x[i, j] + ax[i]  * pxy_x
        ψ_pyy_y[i, j] = (by[i+1] + by[i]) / 2 * ψ_pyy_y[i, j] + (ay[i+1] + ay[i]) / 2  * pyy_y

        vy[i, j] = vy[i, j] + dt / ρ * (pxy_x / κx[i] + ψ_pxy_x[i, j] + pyy_y / κy[j] + ψ_pyy_y[i, j])

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

  heatmap(vy[:, :], framestyle = :box, aspect_ratio = :equal, xlabel = "X", ylabel = "Y", title = "Wave Propagation", color = :seismic, size = (900, 900), clim=(-dt / ρ / 20, dt / ρ / 20))
  
end every 10


