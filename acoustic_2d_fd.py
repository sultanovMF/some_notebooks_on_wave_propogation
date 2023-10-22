import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


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


Cx = (c * dt / dx)**2
Cy = (c * dt / dy)**2
Cz = (c * dt / dy)**2

save_every = 1
snapshots = np.zeros((Nt // save_every + 1, Nx, Ny))

microphone_coord = (Nx // 2, Ny // 2, Nz // 2)
microphone_data = np.zeros(Nt // save_every + 1)


i = 0
for n in range(2, Nt):
    tau = (n - 1) * dt - 0.5

    f[Nx // 2, Ny // 2, Nz // 2] = 10 * (2 / np.sqrt(3) / sigma / np.pi**(0.25) * (1 - (tau/sigma)**2) * np.exp(-tau**2 / 2 / sigma**2))
    #f[Nx // 2, Ny // 2, Nz // 2] = 200 * np.cos(n * 0.15)

    Uxx = U[1, 2:, 1:-1, 1:-1] - 2 * U[1, 1:-1, 1:-1, 1:-1] + U[1, :-2, 1:-1, 1:-1]
    Uyy = U[1, 1:-1, 2:, 1:-1] - 2 * U[1, 1:-1, 1:-1, 1:-1] + U[1, 1:-1, :-2, 1:-1]
    Uzz = U[1, 1:-1, 1:-1, 2:] - 2 * U[1, 1:-1, 1:-1, 1:-1] + U[1, 1:-1, 1:-1, :-2]

    U[2, 1:-1, 1:-1, 1:-1] = 2 * U[1, 1:-1, 1:-1, 1:-1] - U[0, 1:-1, 1:-1, 1:-1] + Cx * Uxx + Cy* Uyy + Cz * Uzz + dt**2 * f[1:-1, 1:-1, 1:-1]
    
    microphone_data[n - 2] = (U[0, *(microphone_coord)])
    
    if (n - 2) % save_every == 0:
        snapshots[i] = U[0, :, :, Nz // 2]  
        i+= 1

    U[0] = U[1]
    U[1] = U[2]

    print(f"{n} / {Nt}",  end="\r")


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