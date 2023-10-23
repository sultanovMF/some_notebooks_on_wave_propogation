# Решение уравнений сесйсмики в средах с затуханием на смещенной сетке

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from numba import jit, njit

lamb = 20000 # Pa
mu = 40000 # Pa
rho = 2700 # kg/m^2

tau_p = 0
tau_s = 0
tau = 10
Lx = 10 #meter
Ly = 10 #meter

Nx = 1000
Ny = 1000

Nt = 200
T = 0.2

dt = T / Nt
dx = Lx / Nx
dy = Ly / Ny

micro_coord = (Nx // 2,   Ny // 2 - 20)
source_coord = (Nx // 2, Ny // 2)
alpha = np.sqrt((lamb + 2 * mu) / rho)

stability_cryteria = np.max([dx, dy]) / (np.sqrt(alpha * (27/24 - 1/24)))
assert dt < stability_cryteria, f"Нет устойчивости! {dt} < {stability_cryteria}"


snap_freq = 1
fM = 40 # Herz

@njit(parallel=True, fastmath = True)
def run():

  vx = np.zeros((Nx, Ny))
  vy = np.zeros((Nx, Ny))

  pxx = np.zeros((Nx, Ny))
  pyy = np.zeros((Nx, Ny))
  pxy = np.zeros((Nx, Ny))

  rxx = np.zeros((Nx, Ny))
  ryy = np.zeros((Nx, Ny))
  rxy = np.zeros((Nx, Ny))


  # 1 / dx * (27/24 * () - 1 / 24 * ())
  p1P = (lamb + 2 * mu) * (1 + tau_p)
  p1S = mu * (1 + tau_s)
  p1d = p1P - 2 * p1S

  p2P = (lamb + 2 * mu) * tau_p
  p2S = mu * tau_s
  p2d = p2P - 2 * p2S


  snapshots = np.zeros((Nt // snap_freq, Nx, Ny))

  ricker_wavelet = lambda t: (1 - 2 * np.pi**2 * fM**2 * t**2) * np.exp(- np.pi**2 * fM**2 * t**2)

  micro_data = np.zeros(Nt // snap_freq)

  snap_idx = 0

  for n in range(0, Nt):
    # print(f"process {n} / {Nt}", end="\r")
    # pxx[*source_coord] = dt * ricker_wavelet(n * dt - 0.1)
    # pyy[*source_coord] = ricker_wavelet(n * dt - 0.1)
    # pxy[*source_coord] = ricker_wavelet(n * dt - 0.1)


    # update velocities
    for i in range(5, Nx - 4):
      for j in range(5, Ny - 4):
        pxx_x = 1 / dx * (27/24 * (pxx[i + 1, j] - pxx[i, j]) - 1 / 24 * (pxx[i + 2, j] - pxx[i - 1, j]))
        pxy_y = 1 / dy * (27/24 * (pxy[i, j] - pxy[i, j - 1]) - 1 / 24 * (pxy[i, j + 1] - pxy[i, j - 2]))
        vx[i, j] = vx[i, j] + dt / rho * (pxx_x + pxy_y)

        pyy_y = 1 / dy * (27/24 * (pyy[i, j + 1] - pyy[i, j]) - 1 / 24 * (pyy[i, j + 2] - pyy[i, j - 1]))
        pxy_x = 1 / dx * (27/24 * (pxy[i, j] - pxy[i - 1, j]) - 1 / 24 * (pxy[i + 1, j] - pxy[i - 2, j]))
        vy[i, j] = vy[i, j] + dt / rho * (pxy_x + pyy_y)
    
    #vx[*source_coord] += dt * np.sin(0.15 * n * dt)
    #vy[*source_coord] += dt * np.sin(0.15 * n * dt)

    vx[*source_coord] += dt * ricker_wavelet(n * dt - 0.05)
    vy[*source_coord] += dt * ricker_wavelet(n * dt - 0.05)

    # update stress
    for i in range(5, Nx - 4):
      for j in range(5, Ny - 4):
        vx_x = 1 / dx * (27/24 * (vx[i, j] - vx[i - 1, j]) - 1 / 24 * (vx[i + 1, j] - vx[i - 2, j]))
        vy_y = 1 / dy * (27/24 * (vy[i, j] - vy[i, j - 1]) - 1 / 24 * (vy[i, j + 1] - vy[i, j - 2]))

        pxx[i, j] = pxx[i, j] + dt * (p1P * vx_x + p1d * vy_y) # + 0.5 * rxx[i, j]  # добавим вторую половинку с нового слоя позже
        pyy[i, j] = pyy[i, j] + dt * (p1d * vx_x + p1P * vy_y) # + 0.5 * ryy[i, j]  # добавим вторую половинку с нового слоя позже

        vx_y = 1 / dy * (27/24 * (vx[i, j + 1] - vx[i, j]) - 1 / 24 * (vx[i, j + 2] - vx[i, j - 1]))
        vy_x = 1 / dx * (27/24 * (vy[i + 1, j] - vy[i, j]) - 1 / 24 * (vy[i + 2, j] - vy[i - 1, j]))

        pxy[i, j] = pxy[i, j] + dt * (p1S * vx_y + p1S * vy_x)  #+ 0.5 * rxy[i, j] # добавим вторую половинку с нового слоя позже

    # update memory variables
    # for i in range(5, Nx - 4):
    #   for j in range(5, Ny - 4):
    #     vx_x = 1 / dx * (27/24 * (vx[i, j] - vx[i - 1, j]) - 1 / 24 * (vx[i + 1, j] - vx[i - 2, j]))
    #     vy_y = 1 / dy * (27/24 * (vy[i, j] - vy[i, j - 1]) - 1 / 24 * (vy[i, j + 1] - vy[i, j - 2]))

    #     rxx[i, j] = 1. / (tau + 1) * (- rxx[i, j] - dt * (p2P * vx_x + p2d * vy_y))
    #     ryy[i, j] = 1. / (tau + 1) * (- ryy[i, j] - dt * (p2d * vx_x + p2P * vy_y))

    #     vx_y = 1 / dy * (27/24 * (vx[i, j + 1] - vx[i, j]) - 1 / 24 * (vx[i, j + 2] - vx[i, j - 1]))
    #     vy_x = 1 / dx * (27/24 * (vy[i + 1, j] - vy[i, j]) - 1 / 24 * (vy[i + 2, j] - vy[i - 1, j]))

    #     rxy[i, j] = 1. / (tau + 1) * (- rxy[i, j] - dt * (p2S * vx_y + p2S * vy_x))

    #     pxx[i, j] += 0.5 * rxx[i, j]
    #     pyy[i, j] += 0.5 * ryy[i, j]
    #     pxy[i, j] += 0.5 * rxy[i, j]

    if (n % snap_freq == 0):
      snapshots[snap_idx] = vx
      micro_data[snap_idx] = vx[*micro_coord] #vx[*micro_coord]
      snap_idx += 1
  
  return snapshots, micro_data

snapshots, micro_data = run()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

frames = []

Umin = np.quantile(snapshots[:, :, :], 0.75)
Umax = np.quantile(snapshots[:, :, :], 0.75)

signal = np.zeros(Nt // snap_freq)
t = np.linspace(0, T, Nt  // snap_freq)

for k in range(Nt  // snap_freq):
    frame = ax1.imshow(snapshots[k, :, :], extent=[0, Lx, 0, Ly])

    frame1, = ax1.plot(micro_coord[0] * dx, micro_coord[1] * dy, 'ro')

    signal, = ax2.plot(t[:k], micro_data[:k], 'b-')

    frames.append([frame, frame1, signal])

ani = animation.ArtistAnimation(fig,frames,interval=10, blit=True, repeat_delay=1000)

#plt.colorbar(frame) 
plt.show()