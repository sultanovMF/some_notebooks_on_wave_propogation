import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

x_range = (0, 1)
Nx = 50
dx = (x_range[1] - x_range[0]) / Nx
x = np.linspace(*x_range, Nx)

y_range = (0, 1)
Ny = 50
dy = (y_range[1] - y_range[0]) / Ny
y = np.linspace(*y_range, Ny)

z_range = (0, 1)
Nz = 50
dz = (z_range[1] - z_range[0]) / Nz
z = np.linspace(zy_range, Nz)

X, Y, Z = np.meshgrid(x, y, z)

c = 1
s = lambda t, x: 0

t_range = (0, 1)
dt = 1 / c / (1 / dx + 1 / dy)
Nt = int((t_range[1] - t_range[0]) / dt)
t = np.linspace(*t_range, Nt)

assert c * dt * (1 / dx + 1 / dy + 1 / dx) <= 1, f"Условие куранта не выполнено! ( {c * dt * (1 / dx + 1 / dy + 1 / dz)} )"

print("Решение уравнения u_tt = c^2 u_xx + s(t, x)")
print("Параметры модели:")
print(f"\t c = {c}, s(t, x) = 0")
print("Параметры сетки:")
print(f"\t dx = {dx} Nx = {Nx}")
print(f"\t dy = {dt} Ny = {Ny}")
print(f"\t dz = {dz} Nz = {Nz}")
print(f"\t dt = {dt} Nt = {Nt}")

# f = np.zeros()
U = np.zeros((Nt, Nx, Ny, Nz))
f = np.zeros((Nx, Ny, Nz))

sigma = 0.1
ricker = lambda x, y: 1 / np.pi / sigma**4 * (1 - 0.5 * (x**2 + y**2) / sigma**2) * np.exp(- (x**2 + y**2) / 2 / sigma**2)


# U[0, :, :] = ricker(X - 0.5, Y - 0.5)
# U[1, 1:-1, 1:-1] =  U[0, 1:-1, 1:-1] + \
#         Cx * (U[0, 2:, 1:-1] - 2 * U[0, 1:-1, 1:-1] + U[0, :-2, 1:-1]) + \
#         Cy * (U[0, 1:-1, 2:] - 2 * U[0, 1:-1, 1:-1] + U[0, 1:-1, :-2]) + \
#         dt**2 * f[1:-1, 1:-1]

microphone_coord = (Nx // 2, Ny // 2)
microphone_data = np.zeros(Nt)
fs = np.zeros(Nt)

Cx = (c * dt / dx)**2
Cy = (c * dt / dy)**2

for n in range(2, Nt):
    tau = (n - 1) * dt - 0.5

    f[Nx // 2, Ny // 2] = 2 / np.sqrt(3) / sigma / np.pi**(0.25) * (1 - (tau/sigma)**2) * np.exp(-tau**2 / 2 / sigma**2)

    # for i in range (1, Nx - 1):
    #     for j in range(1, Ny - 1):
    #         Uxx = U[n - 1, i + 1, j] - 2 * U[n - 1, i, j] + U[n - 1, i - 1, j]
    #         Uyy = U[n - 1, i, j + 1] - 2 * U[n - 1, i, j] + U[n - 1, i, j - 1]

    #         U[n, i, j] = - U[n - 2, i, j] + 2 * U[n - 1, i, j] + Cx * Uxx + Cy * Uyy + dt**2 * f[i, j]

    # U[:, Nx // 2, Ny // 2] = 20 * np.cos(n * 0.15)
    U[n, 1:-1, 1:-1] = 2 * U[n - 1, 1:-1, 1:-1] - U[n - 2, 1:-1, 1:-1] + \
        Cx**2 * (U[n - 1, 2:, 1:-1] - 2 * U[n - 1, 1:-1, 1:-1] + U[n - 1, :-2, 1:-1]) + \
        Cy**2 * (U[n - 1, 1:-1, 2:] - 2 * U[n - 1, 1:-1, 1:-1] + U[n - 1, 1:-1, :-2]) + \
        dt**2 * f[1:-1, 1:-1]
    
    microphone_data[n] = (U[n, *(microphone_coord)])


    print(f"{n} / {Nt}",  end="\r")


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

frames = []

Umin = np.min(U[:, :, :])
Umax = np.max(U[:, :, :])

signal = np.zeros(Nt)

for k in range(Nt):
    frame = ax1.imshow(U[k, :,:], vmin=Umin, vmax=Umax, extent=[x.min(), x.max(), y.min(), y.max()])
    frame1, = ax1.plot(microphone_coord[0] * dx, microphone_coord[1] * dy, 'ro')

    signal, = ax2.plot(t[:k], microphone_data[:k], 'b-')

    frames.append([frame, frame1, signal])

ani = animation.ArtistAnimation(fig,frames,interval=10, blit=True, repeat_delay=1000)

plt.colorbar(frame) 
plt.show()