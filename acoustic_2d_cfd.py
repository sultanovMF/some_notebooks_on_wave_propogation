import time
import datetime

from numba import jit 

import numpy as np
from scipy.linalg import solve_banded
from scipy.sparse import csr_matrix

import matplotlib.pyplot as plt
import matplotlib.animation as animation

@jit(nopython=True, fastmath = True)
def TDMA(a,b,c,d):
  n = len(d)
  x = np.empty(n)
  w = np.empty(n)
  bc = np.empty(n)
  dc = np.empty(n)
  bc[0] = b[0]
  dc[0] = d[0]
  for i in range(1,n):
    w[i] = a[i-1]/bc[i-1]
    bc[i] = b[i] - w[i]*c[i-1]
    dc[i] = d[i] - w[i]*dc[i-1]

  x[n-1] = dc[n-1]/bc[n-1]
  for k in range(n-2,-1,-1):
    x[k] = (dc[k]-c[k]*x[k+1])/bc[k]
  return x

x_range = (0, 1)
Nx = 100
dx = (x_range[1] - x_range[0]) / Nx
x = np.linspace(*x_range, Nx)

y_range = (0, 1)
Ny = 100
dy = (y_range[1] - y_range[0]) / Ny
y = np.linspace(*y_range, Ny)

z_range = (0, 1)
Nz = 100
dz = (z_range[1] - z_range[0]) / Nz
z = np.linspace(z_range, Nz)


c = 1
s = lambda t, x: 0

t_range = (0, 3)
dt = 1 / c / (1 / dx + 1 / dy + 1 / dz)
Nt = int((t_range[1] - t_range[0]) / dt)

assert c * dt * (1 / dx + 1 / dy + 1 / dx) <= 1, f"Условие куранта не выполнено! ( {c * dt * (1 / dx + 1 / dy + 1 / dz)} )"

print("Решение уравнения u_tt = c^2 u_xx + s(t, x)")
print("Параметры модели:")
print(f"\t c = {c}, s(t, x) = 0")
print("Параметры сетки:")
print(f"\t dx = {dx} Nx = {Nx}")
print(f"\t dy = {dy} Ny = {Ny}")
print(f"\t dz = {dz} Nz = {Nz}")
print(f"\t dt = {dt} Nt = {Nt}")

# f = np.zeros()
U = np.zeros((3, Nx, Ny, Nz))
f = np.zeros((Nx, Ny, Nz))

sigma = 0.1

save_every = 1
snapshots = np.zeros((Nt // save_every + 1, Nx, Ny))

microphone_coord = (Nx // 2, Ny // 2, Nz // 2)
microphone_data = np.zeros(Nt // save_every + 1)

idx = 0

err = []



a0 = 1
a1 = 1/10
b0 = -12/(5*dx**2)
b1 = 6/(5*dx**2)

# Ax = a0*np.eye(Nx)+a1*np.eye(Nx,k=1)+a1*np.eye(Nx,k=-1)
Bx = b0*np.eye(Nx)+b1*np.eye(Nx,k=1)+b1*np.eye(Nx,k=-1)

# Ay = a0*np.eye(Ny)+a1*np.eye(Ny,k=1)+a1*np.eye(Ny,k=-1)
By = b0*np.eye(Ny)+b1*np.eye(Ny,k=1)+b1*np.eye(Ny,k=-1)

# Az = a0*np.eye(Nz)+a1*np.eye(Nz,k=1)+a1*np.eye(Nz,k=-1)
Bz = b0*np.eye(Nz)+b1*np.eye(Nz,k=1)+b1*np.eye(Nz,k=-1)

Bx = csr_matrix(Bx)
By = csr_matrix(By)
Bz = csr_matrix(Bz)

a0x = a0*np.ones(Nx)
a1x = a1*np.ones(Nx)

a0y = a0*np.ones(Ny)
a1y = a1*np.ones(Ny)

a0z = a0*np.ones(Nz)
a1z = a1*np.ones(Nz)

u_xx = np.zeros((Nx,Ny,Nz))
u_yy = np.zeros((Nx,Ny,Nz))
u_zz = np.zeros((Nx,Ny,Nz))
f = np.zeros((Nx,Ny,Nz))
u = np.zeros((Nx,Ny,Nz))

kappa = c * dt / dx

for n in range(2, Nt):

    if n % save_every == 0:
        now = datetime.datetime.now()
        print(' Step ',n,'/',Nt,' @ ',now.strftime("%H:%M:%S"), end='\r')
    tau = (n - 1) * dt - 0.5

    #f[Nx // 2, Ny // 2, Nz // 2] = 10 * (2 / np.sqrt(3) / sigma / np.pi**(0.25) * (1 - (tau/sigma)**2) * np.exp(-tau**2 / 2 / sigma**2))
    f[Nx // 2, Ny // 2, Nz // 2] = 200 * np.cos(n * 0.15)

    for j in range(0,Ny):
        for k in range(0,Nz):
            u_xx[:,j,k] = TDMA(a1x,a0x,a1x,Bx @ U[1, :, j, k])
    
    for i in range(0,Nx):
        for k in range(0,Nz):
            u_yy[i,:,k] = TDMA(a1y,a0y,a1y,By @ U[1, i, :, k])
    
    for i in range(0,Nx):
        for j in range(0,Ny):
            u_zz[i,j,:] = TDMA(a1z,a0z,a1z,Bz @ U[1, i, j, :])

    U[2, 1:-1, 1:-1, 1:-1] = 2 * U[1, 1:-1, 1:-1, 1:-1] - U[0, 1:-1, 1:-1, 1:-1]  \
        + dt**2 * ( c**2 * u_xx[1:-1, 1:-1, 1:-1] + c**2 * u_yy[1:-1, 1:-1, 1:-1] + c**2 * u_zz[1:-1, 1:-1, 1:-1] + f[1:-1, 1:-1, 1:-1] )
    
    # Mur boundary
    U[2, 0, :, :] = U[1, 1, :, :] + (kappa - 1) / (kappa + 1) * (U[2, 1, :, :] - U[1, 0, :, :])
    U[2, :, 0, :] = U[1, :, 1, :] + (kappa - 1) / (kappa + 1) * (U[2, :, 1, :] - U[1, :, 0, :])
    U[2, :, :, 0] = U[1, :, :, 1] + (kappa - 1) / (kappa + 1) * (U[2, :, :, 1] - U[1, :, :, 0])

    U[2, Nx - 1, :, :] = U[1, Nx - 2, :, :] + (kappa - 1) / (kappa + 1) * (U[2, Nx - 2, :, :] - U[1, Nx - 1, :, :])
    U[2, :, Ny - 1, :] = U[1, :, Ny - 2, :] + (kappa - 1) / (kappa + 1) * (U[2, :, Ny - 2, :] - U[1, :, Ny - 1, :])
    U[2, :, :, Nz - 1] = U[1, :, :, Nz - 2] + (kappa - 1) / (kappa + 1) * (U[2, :, :, Nz - 2] - U[1, :, :, Nz - 1])
    
    if (n - 2) % save_every == 0:
        snapshots[idx] = U[0, Nx // 2, :, :]  
        microphone_data[idx] = (U[2, *(microphone_coord)])
        idx += 1

    U[0] = U[1]
    U[1] = U[2]



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

frames = []

Umin = np.min(snapshots[:, :, :])
Umax = np.max(snapshots[:, :, :])

signal = np.zeros(Nt)
t = np.linspace(*t_range, Nt // save_every + 1)


for k in range(Nt // save_every + 1):
    frame = ax1.imshow(snapshots[k, :, :], vmin=Umin, vmax=Umax, extent=[x.min(), x.max(), y.min(), y.max()])
    frame1, = ax1.plot(microphone_coord[0] * dx, microphone_coord[1] * dy, 'ro')

    signal, = ax2.plot(t[:k], microphone_data[:k], 'b-')

    frames.append([frame, frame1, signal])

ani = animation.ArtistAnimation(fig,frames,interval=10, blit=True, repeat_delay=1000)

plt.colorbar(frame) 
plt.show()