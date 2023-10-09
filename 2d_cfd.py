import numpy as np
from scipy.sparse.linalg import cgs
from scipy.sparse import csc_matrix

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm

c = 2.

x_range = (0, 1)
y_range = (0, 1)
t_range = (0, 1)

Nx = 100
Ny = 100
Nt = 300

center = ((Nx  - 2) // 2, (Ny - 2) // 2)

dx = (x_range[1] - x_range[0]) / Nx
dy = (y_range[1] - y_range[0]) / Ny
# dt = (t_range[1] - t_range[0]) / Nt
dt = 0.95 / (1 / dx + 1 / dy) / c

assert c * dt * (1 / dx + 1 / dy) <= 1, f"Условие куранта не выполнено! ( {c * dt * (1 / dx + 1 / dy)} )"

x = np.linspace(*x_range, Nx)
y = np.linspace(*y_range, Ny)

X, Y = np.meshgrid(x, y)
U = np.zeros((Nt, Nx, Ny))

a_0 = -2.4
a_1 = 1.2 # a_-1 = a1

b_0 = 1
b_1 = 0.1

Bx = csc_matrix(b_0 * np.eye(Nx - 2, Nx - 2) + b_1 * np.eye(Nx - 2, Nx - 2, 1) +  b_1 * np.eye(Nx - 2, Nx - 2, -1))
Ax = csc_matrix(a_0 * np.eye(Nx - 2, Nx - 2) + a_1 * np.eye(Nx - 2, Nx - 2, 1) +  a_1 * np.eye(Nx - 2, Nx - 2, -1))

By = csc_matrix(b_0 * np.eye(Ny - 2, Ny - 2) + b_1 * np.eye(Ny - 2, Ny - 2, 1) +  b_1 * np.eye(Ny - 2, Ny - 2, -1))
Ay = csc_matrix(a_0 * np.eye(Ny - 2, Ny - 2) + a_1 * np.eye(Ny - 2, Ny - 2, 1) +  a_1 * np.eye(Ny - 2, Ny - 2, -1))


f = np.zeros((Nx - 2,Ny -2))

Uxx = np.zeros((Nx-2, Ny-2))
Uyy = np.zeros((Nx-2, Ny-2))


for i in range(2, Nt):
    # for j in range(1, Ny-2):
    #     Uxx[:, j], _ = cgs(Bx, 1 / dx**2 * Ax @ U[i-1, 1:-1, j])
    # for j in range(1, Ny-2):
    #     Uyy[j, :], _ = cgs(By, 1 / dy**2 * Ay @ U[i-1, j, 1:-1])

    #f[*center] = 

    #U[i, 1:-1, 1:-1] = 2 * U[i-1, 1:-1, 1:-1] - U[i-2, 1:-1, 1:-1] + dt**2 * (c**2 * (Uxx + Uyy) + f)
    #U[0:2, *(center)] = np.cos(i * 0.25) * 10

    U[i, 1:-1, 1:-1] = 2 * U[i-1, 1:-1, 1:-1] - U[i-2, 1:-1, 1:-1] + dt**2 * \
          (c**2 * (1 / dx**2 * (U[i-1, 2:, 1:-1] - 2 * U[i-1, 1:-1, 1:-1] + U[i-1, 0:-2, 1:-1]) \
            + 1 / dy**2 * (U[i-1, 1:-1, 2:] - 2 * U[i-1, 1:-1, 1:-1] + U[i-1, 1:-1, 0:-2])) \
            + f)
        
    
    print(i)


fig, ax = plt.subplots()
pcm = ax.pcolormesh(X, Y, U[0]) #, cmap = cm.gray
plt.colorbar(pcm, ax=ax)  # Add a colorbar

def update(frame):
    pcm.set_array(U[frame % Nt].ravel())
    return pcm,

ani = animation.FuncAnimation(fig, update, frames=100, interval=0.1)

plt.show()



