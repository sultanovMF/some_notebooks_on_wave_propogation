import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.sparse.linalg import cgs
from scipy.sparse import csc_matrix

x_range = (0, 1)
Nx = 250
dx = (x_range[1] - x_range[0]) / Nx
x = np.linspace(*x_range, Nx)

y_range = (0, 1)
Ny = 250
dy = (y_range[1] - y_range[0]) / Ny
y = np.linspace(*y_range, Ny)

X, Y = np.meshgrid(x, y)

c = 1
s = lambda t, x: 0

t_range = (0, 0.5)
dt = 1 / c / (1 / dx + 1 / dy)
Nt = int((t_range[1] - t_range[0]) / dt)

assert c * dt * (1 / dx + 1 / dy) <= 1, f"Условие куранта не выполнено! ( {c * dt * (1 / dx + 1 / dy)} )"

print("Решение уравнения u_tt = c^2 u_xx + s(t, x)")
print("Параметры модели:")
print(f"\t c = {c}, s(t, x) = 0")
print("Параметры сетки:")
print(f"\t dx = {dx} Nx = {Nx}")
print(f"\t dy = {dt} Ny = {Ny}")
print(f"\t dt = {dt} Nt = {Nt}")

# f = np.zeros()
U = np.zeros((Nt, Nx, Ny))

Cx = c**2 * dt**2 / dx**2
Cy = c**2 * dt**2 / dx**2

f = np.zeros((Nx, Ny))

sigma = 0.01
ricker = lambda x, y: 1 / np.pi / sigma**4 * (1 - 0.5 * (x**2 + y**2) / sigma**2) * np.exp(- (x**2 + y**2) / 2 / sigma**2)

microphone_place = (Nx // 2, Ny // 2)
microphone_data = []

# a_0 = -2.4
# a_1 = 1.2 # a_-1 = a1

# b_0 = 1
# b_1 = 0.1

# Bx = csc_matrix(b_0 * np.eye(Nx - 2, Nx - 2) + b_1 * np.eye(Nx - 2, Nx - 2, 1) +  b_1 * np.eye(Nx - 2, Nx - 2, -1))
# Ax = csc_matrix(a_0 * np.eye(Nx - 2, Nx - 2) + a_1 * np.eye(Nx - 2, Nx - 2, 1) +  a_1 * np.eye(Nx - 2, Nx - 2, -1))

# By = csc_matrix(b_0 * np.eye(Ny - 2, Ny - 2) + b_1 * np.eye(Ny - 2, Ny - 2, 1) +  b_1 * np.eye(Ny - 2, Ny - 2, -1))
# Ay = csc_matrix(a_0 * np.eye(Ny - 2, Ny - 2) + a_1 * np.eye(Ny - 2, Ny - 2, 1) +  a_1 * np.eye(Ny - 2, Ny - 2, -1))

# Uxx = np.zeros((Nx-2, Ny-2))
# Uyy = np.zeros((Nx-2, Ny-2))

for n in range(Nt):
    tau = n * dt - 0.25
    f[Nx // 2, Ny // 2] = 2 / np.sqrt(3) / sigma / np.pi**(0.25) * (1 - (tau/sigma)**2) * np.exp(-tau**2 / 2 / sigma**2)

    U[n, 1:-1, 1:-1] = 2 * U[n - 1, 1:-1, 1:-1] - U[n - 2, 1:-1, 1:-1] + \
        Cx * (U[n - 1, 2:, 1:-1] - 2 * U[n - 1, 1:-1, 1:-1] + U[n - 1, :-2, 1:-1]) + \
        Cy * (U[n - 1, 1:-1, 2:] - 2 * U[n - 1, 1:-1, 1:-1] + U[n - 1, 1:-1, :-2]) + \
        dt**2 * f[1:-1, 1:-1]

    # for j in range(1, Ny-2):
    #     Uxx[:, j], _ = cgs(Bx, 1 / dx**2 * Ax @ U[n-1, 1:-1, j])
    # for j in range(1, Nx-2):
    #     Uyy[j, :], _ = cgs(By, 1 / dy**2 * Ay @ U[n-1, j, 1:-1])

    # U[n, 1:-1, 1:-1] = 2 * U[n - 1, 1:-1, 1:-1] - U[n - 2, 1:-1, 1:-1] + \
    #     c**2 * dt**2 * (Uxx) + \
    #     c**2 * dt**2 * (Uyy) + \
    #     dt**2 * f[1:-1, 1:-1]
    
    # microphone_data.append(f[Nx // 2, Ny // 2])
    # microphone_data.append(U[n, *(microphone_place)])

    print(f"{n} / {Nt}")

# plt.plot(microphone_data)
# plt.show()

fig = plt.figure(1,(16,8))
ax = fig.add_subplot()
frames = []

Umin = np.min(U[:, :, :])
Umax = np.max(U[:, :, :])

for k in range(Nt):
    frame = ax.imshow(U[k, :,:], vmin=Umin, vmax=Umax, extent=[x.min(), x.max(), y.min(), y.max()])
    frames.append([frame])


ani = animation.ArtistAnimation(fig,frames,interval=50,
                         blit=True,repeat_delay=1000)

plt.colorbar(frame) 
plt.show()