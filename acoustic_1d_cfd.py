import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from scipy.sparse.linalg import cgs
from scipy.sparse import csc_matrix

# Решение задачи u_xx = c^2 u_tt + s(x, t)
s = lambda x, t: 0 
# начальные условия
f = lambda x: (8*x*(1-x)**2) # u(x, 0)
g = lambda x: 0 # u_t(x, 0)
# граничные условия
phi = lambda t: 0 # left
psi = lambda t: 0 # right

c = 2.

# Настройки сетки
x_range = (0, 1)
t_range = (0, 1)

Nx = 10000
Nt = 30000

dx = (x_range[1] - x_range[0]) / Nx
dt = (t_range[1] - t_range[0]) / Nt

# assert c * dt / dx <= 1, f"Условие куранта не выполнено! ( {c * dt / dx} )"

x = np.linspace(*x_range, Nx)
t = np.linspace(*t_range, Nt)

# Настройка параметров компактной сетки
a_0 = -2.4
a_1 = 1.2 # a_-1 = a1

b_0 = 1
b_1 = 0.1

B =  csc_matrix(b_0 * np.eye(Nx - 2, Nx - 2) + b_1 * np.eye(Nx - 2, Nx - 2, 1) +  b_1 * np.eye(Nx - 2, Nx - 2, -1))
A =  csc_matrix(a_0 * np.eye(Nx - 2, Nx - 2) + a_1 * np.eye(Nx - 2, Nx - 2, 1) +  a_1 * np.eye(Nx - 2, Nx - 2, -1))

# Решение
u = np.zeros((Nt, Nx))

# def u_xx(u, i):
#     w_xx = np.zeros(Nx - 2)
#     w_xx[0]  = 3.75 * u[i, 0]  - 12.8333 * u[i, 1]  + 17.8333 * u[i, 2] - 13 * u[i, 3]  + 5.0833 * u[i, 4]  - 0.8333 * u[i, 5] 
#     w_xx[-1] = 3.75 * u[i, -6] - 12.8333 * u[i, -5] + 17.8333 * u[i, -4] -13 * u[i, -3] + 5.0833 * u[i, -2] - 0.8333 * u[i, -1] 

#     w = np.zeros(Nx - 2)
#     w[0] = 6 / 5 * u[0]
#     w[-1] = 6 / 5 * u[-1]

#     u_xx = solve(B, 1 / dx**2 (A @ u[1:-1] + w) - w_xx, assume_a='pos')
#     return u_xx

# первый слой
u[0, :] = f(x[:])
# спец формула для второго слоя
uxx, _ = cgs(B, A @ u[0, 1:-1])
u[1, 1:-1] = u[0, 1:-1] + dt**2 / 2 * (c**2 * uxx  + s(x, 0))
u[1, 0]  = phi(dt)
u[1, -1] = psi(dt)

for i in range(2, Nt):
    # print(i, np.max(np.abs(u_xx(u, i - 1) - (u[i-1, 2:] - 2 * u[i-1, 1:-1] + u[i-1, 0:-2]))))
    #u[i, 1:-1] = 2 * u[i-1, 1:-1] - u[i-2, 1:-1] + c**2 * dt**2 / dx**2 * (u[i-1, 2:] - 2 * u[i-1, 1:-1] + u[i-1, 0:-2]) + dt**2 * s(x[1:-1], i * dt)
    uxx, _ = cgs(B, 1 / dx**2 * A @ u[i - 1, 1:-1])

    u[i, 1:-1] = 2 * u[i - 1, 1:-1] - u[i - 2, 1:-1] \
        + dt**2 * (c**2 * uxx + s(x, (i - 1) * dt))
    u[i, 0]  = phi(i * dt)
    u[i, -1] = psi(i * dt)

    # print(np.max(np.abs((u[i] - u[i-1]))))

u_exact = lambda x, t: np.sum([32 * ((-1) ** n + 2) * np.sin(n * np.pi * x) * np.cos(2 * n * np.pi * t) / np.pi ** 3 / n ** 3 for n in range(1, 16)], axis=0)

err = []
for i in range(0, Nt):
    e = np.max(np.abs(u_exact(x, i * dt) - u[i, :]))
    err.append(e)

plt.plot(err, '-ok')
plt.show()

# fig, ax = plt.subplots()
# line, = ax.plot(x, u[0, :])
# #line, = ax.plot(x, target(x, 0))

# def init():  # only required for blitting to give a clean slate.
#     line.set_ydata([np.nan] * len(x))
#     return line,

# def animate(i):
#     line.set_ydata(u[i % Nt, :])  # update the data.
#     #line.set_ydata(target(x, (i % Nt) * dt ))  # update the data.
#     return line,

# ani = animation.FuncAnimation(
#     fig, animate, init_func=init, interval=10, repeat=True, blit=True, save_count=Nt)

# plt.ylim((-2, 2))
# plt.show()


# # To save the animation, use e.g.
# #
# # ani.save("movie.mp4")
# #
# # or
# #
# # from matplotlib.animation import FFMpegWriter
# # writer = FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
# # ani.save("movie.mp4", writer=writer)
