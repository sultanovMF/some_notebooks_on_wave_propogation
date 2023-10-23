using Plots;
using Base.Threads;
using LoopVectorization;

Threads.nthreads() = 8


# viscoelastic params
τP = 0
τS = 0
τ = 10

# lame params
μ = 40000 #5.2e9 # 
λ = 20000 #5.0e9 # Pa

# density
ρ = 2700 #2000 # kg / m^3

# Настройки сетки
Lx = 1 # m
Ly = 1 # m
Lz = 1 # m

Nx = 200 # points
Ny = 200 # points
Nz = 200 # points

dx = Lx / Nx # m
dy = Ly / Ny # m
dz = Lz / Nz # m

# Настройки временной сетки
T  = 0.1 # sec
Nt = 1000
dt = T / Nt

fm = 40 # Hz, dominant frequency in ricker wavelet

M = 6 # order of approx
α = [1.221336364746061, -0.0969314575195135, 0.017447662353510373, -0.0029672895159030804, 0.00035900539822037244, -2.1847811612209015e-5] # spacial finite difference coefficient

vx = zeros(Nx, Ny, Nz)
vy = zeros(Nx, Ny, Nz)
vz = zeros(Nx, Ny, Nz)

pxx = zeros(Nx, Ny, Nz)
pyy = zeros(Nx, Ny, Nz)
pzz = zeros(Nx, Ny, Nz)
pxy = zeros(Nx, Ny, Nz)
pxz = zeros(Nx, Ny, Nz)
pyz = zeros(Nx, Ny, Nz)

rxx = zeros(Nx, Ny, Nz)
ryy = zeros(Nx, Ny, Nz)
rzz = zeros(Nx, Ny, Nz)
rxy = zeros(Nx, Ny, Nz)
rxz = zeros(Nx, Ny, Nz)
ryz = zeros(Nx, Ny, Nz)

p1P = (λ + 2 * μ) * (1 + τP)
p1S = μ * (1 + τS)
p1d = p1P - 2 * p1S

p2P = (λ + 2 * μ) * τP
p2S = μ * τS
p2d = p2P - 2 * p2S

σxx_x = 0
σxy_y = 0
σxz_z = 0

println("Nt = ", Nt)

ricker_wavelet(t) = (1. - 2 * pi^2 * fm^2 * t^2) * exp(-pi^2 * fm^2 * t^2)

xs = range(0, stop=Lx, length=Nx)
ys = range(0, stop=Ly, length=Ny)

micro = zeros(Nt)

for n in 1:Nt
  println(n, "/", Nt)
  
  # initial source
  pxx[Nx ÷ 2, Ny ÷ 2, Nz ÷ 2] += dt * ricker_wavelet(dt * n - 0.01)
  # pyy[Nx ÷ 2, Ny ÷ 2, Nz ÷ 2] += dt * ricker_wavelet(dt * n)
  # pzz[Nx ÷ 2, Ny ÷ 2, Nz ÷ 2] += dt * ricker_wavelet(dt * n)

  # update velocities
  @tturbo for i in M+1:Nx-M-1
    for j in M+1:Ny-M-1
      for k in M+1:Nz-M-1
        pxx_x = 0.0
        pxy_y = 0.0
        pxz_z = 0.0

        pxy_x = 0.0
        pyy_y = 0.0
        pyz_z = 0.0

        pxz_x = 0.0
        pyz_y = 0.0
        pzz_z = 0.0
        
        for m in 1:M
          pxx_x += α[m] * (pxx[i + m, j, k] - pxx[i - m + 1, j, k])
          pxy_y += α[m] * (pxy[i, j + m - 1, k] - pxy[i, j - m, k])
          pxz_z += α[m] * (pxy[i, j, k + m - 1] - pxy[i, j, k - m])

          pxy_x += α[m] * (pxy[i + m - 1, j, k] - pxy[i - m, j, k])
          pyy_y += α[m] * (pyy[i, j + m, k] - pyy[i, j - m + 1, k])
          pyz_z += α[m] * (pyz[i, j, k + m - 1] - pyz[i, j, k - m])

          pxz_x += α[m] * (pxz[i + m - 1, j, k] - pxz[i - m, j, k])
          pyz_y += α[m] * (pyz[i, j + m - 1, k] - pyz[i, j - m, k])
          pzz_z += α[m] * (pzz[i, j, k + m] - pzz[i, j, k - m + 1])
        end

        vx[i, j, k] = vx[i, j, k] + dt / ρ * (pxx_x / dx + pxy_y /dy + pxz_z / dz)
        vy[i, j, k] = vy[i, j, k] + dt / ρ * (pxy_x / dx + pyy_y /dy + pyz_z / dz)
        vz[i, j, k] = vy[i, j, k] + dt / ρ * (pxz_x / dx + pyz_y /dy + pzz_z / dz)
      end
    end
  end

  
  #update stress
  @tturbo for i in M+1:Nx-M-1
    for j in M+1:Ny-M-1
      for k in M+1:Nz-M-1
        vx_x = 0 
        vy_y = 0 
        vz_z = 0 
        
        vy_z = 0 
        vz_y = 0 
      
        vz_x = 0 
        vx_z = 0
      
        vx_y = 0
        vy_x = 0 

        for m in 1:M
          vx_x += α[m] * (vx[i + m - 1, j, k] - vx[i - m, j, k])
          vy_y += α[m] * (vy[i, j + m - 1, k] - vy[i, j - m, k])
          vz_z += α[m] * (vz[i, j, k + m - 1] - vz[i, j, k - m])

          vy_z += α[m] * (vy[i, j, k + m] - vy[i, j, k - m + 1])
          vy_x += α[m] * (vy[i + m, j, k] - vy[i - m + 1, j, k])

          vz_y += α[m] * (vz[i, j + m, k] - vz[i, j - m + 1, k])
          vz_x += α[m] * (vz[i + m, j, k] - vz[i - m + 1, j, k])

          vx_z += α[m] * (vx[i, j, k + m] - vx[i, j, k - m + 1])
          vx_y += α[m] * (vx[i, j + m, k] - vx[i, j - m + 1, k])
        end

        pxx[i, j, k] = pxx[i, j, k] + dt * (p1P * vx_x / dx + p1d * vy_y /dy + p1d * vz_z / dz) # + 0.5 * rxx[i, j, k])
        pyy[i, j, k] = pyy[i, j, k] + dt * (p1d * vx_x / dx + p1P * vy_y /dy + p1d * vz_z / dz) # + 0.5 * ryy[i, j, k])
        pzz[i, j, k] = pzz[i, j, k] + dt * (p1d * vx_x / dx + p1d * vy_y /dy + p1P * vz_z / dz) # + 0.5 * rzz[i, j, k])

        pyz[i, j, k] = pyz[i, j, k] + dt * (p1S * vy_z + p1S * vz_y) # + 0.5 * rxx[i, j, k])
        pxz[i, j, k] = pxz[i, j, k] + dt * (p1S * vz_x + p1S * vx_z) # + 0.5 * rxz[i, j, k])
        pxy[i, j, k] = pxy[i, j, k] + dt * (p1S * vx_y + p1S * vy_x) # + 0.5 * rxy[i, j, k])
      end
    end
  end

  # update memory variables
  # @tturbo for i in 5:Nx-5
  #   for j in 5:Ny-5
  #     for k in 5:Nz-5
  #       vx_x = 0 
  #       vy_y = 0 
  #       vz_z = 0 
        
  #       vy_z = 0 
  #       vz_y = 0 
      
  #       vz_x = 0 
  #       vx_z = 0
      
  #       vx_y = 0
  #       vy_x = 0 

  #       for m in 1:M
  #         vx_x += α[m] * (vx[i + m - 1, j, k] - vx[i - m, j, k])
  #         vy_y += α[m] * (vy[i, j + m - 1, k] - vy[i, j - m, k])
  #         vz_z += α[m] * (vz[i, j, k + m - 1] - vz[i, j, k - m])

  #         vy_z += α[m] * (vy[i, j, k + m] - vy[i, j, k - m + 1])
  #         vy_x += α[m] * (vy[i + m, j, k] - vy[i - m + 1, j, k])

  #         vz_y += α[m] * (vz[i, j + m, k] - vz[i, j - m + 1, k])
  #         vz_x += α[m] * (vz[i + m, j, k] - vz[i - m + 1, j, k])

  #         vx_z += α[m] * (vx[i, j, k + m] - vx[i, j, k - m + 1])
  #         vx_y += α[m] * (vx[i, j + m, k] - vx[i, j - m + 1, k])
  #       end

  #       rxx[i, j, k] = 1. / (τ + 0.5 * dt) * ( rxx[i, j, k] - dt * (p2P * vx_x / dx + p2d * vy_y /dy + p2d * vz_z / dz + 0.5 * rxx[i, j, k]) )
  #       ryy[i, j, k] = 1. / (τ + 0.5 * dt) * ( ryy[i, j, k] - dt * (p2d * vx_x / dx + p2P * vy_y /dy + p2d * vz_z / dz + 0.5 * ryy[i, j, k]) )
  #       rzz[i, j, k] = 1. / (τ + 0.5 * dt) * ( rzz[i, j, k] - dt * (p2d * vx_x / dx + p2d * vy_y /dy + p2P * vz_z / dz + 0.5 * rzz[i, j, k]) )
  #       ryz[i, j, k] = 1. / (τ + 0.5 * dt) * ( ryz[i, j, k] - dt * (p2S * vy_z + p2S * vz_y + 0.5 * rxx[i, j, k]) )
  #       rxz[i, j, k] = 1. / (τ + 0.5 * dt) * ( rxz[i, j, k] - dt * (p2S * vz_x + p2S * vx_z + 0.5 * rxz[i, j, k]) )
  #       rxy[i, j, k] = 1. / (τ + 0.5 * dt) * ( rxy[i, j, k] - dt * (p2S * vx_y + p2S * vy_x + 0.5 * rxy[i, j, k]) )
  #     end
  #   end
  # end

  #surface(xs, ys, pxx[:, :, Nz ÷ 2])
  #plot(pxx[Nx ÷ 2, Ny ÷ 2, Nz ÷ 2])

  micro[n] = pxx[Nx ÷ 2, Ny ÷ 2, Nz ÷ 2]
end #every 10


plot(micro)