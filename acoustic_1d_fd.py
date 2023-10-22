import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

c = 2.

x_range = (0, 1)
t_range = (0, 1)

Nx = 1000
Nt = 3000

dx = (x_range[1] - x_range[0]) / Nx
dt = (t_range[1] - t_range[0]) / Nt

assert c * dt / dx <= 1, f"Условие куранта не выполнено! ( {c * dt / dx} )"

x = np.linspace(*x_range, Nx)
t = np.linspace(*t_range, Nt)

# начальные условия
f = lambda x: (8*x*(1-x)**2)
g = lambda x: 0
s = lambda x, t: 0
A = lambda t: 0
B = lambda t: 0

u = np.zeros((Nt, Nx))

u[0, :] = f(x[:])
u[1, 1:-1] = u[0, 1:-1] + g(x[1:-1]) * dt + c**2 * dt**2 / 2 / dx**2 * (u[0, 2:] - 2 * u[0, 1:-1] + u[0, :-2]) + s(x[1:-1], dt)
u[1, 0] = A(dt)
u[1, -1] = B(dt)

for i in range(2, Nt):
    u[i, 1:-1] = 2 * u[i-1, 1:-1] - u[i-2, 1:-1] + c**2 * dt**2 / dx**2 * (u[i-1, 2:] - 2 * u[i-1, 1:-1] + u[i-1, 0:-2]) + dt**2 * s(x[1:-1], i * dt)
    u[i, 0] = A(i * dt)
    u[i, -1] = B(i * dt)

#target = lambda x, t: np.sum([32 * ((-1) ** n + 2) * np.sin(n * np.pi * x) * np.cos(2 * n * np.pi * t) / np.pi ** 3 / n ** 3 for n in range(1, 16)])

target = lambda x, t: np.sum([32 * ((-1) ** n + 2) * np.sin(n * np.pi * x) * np.cos(2 * n * np.pi * t) / np.pi ** 3 / n ** 3 for n in range(1, 16)], axis=0)

err = []
for i in range(0, Nt):
    err.append(np.max(np.abs(target(x, i * dt) - u[i, :])))

print(np.max(err))
plt.plot(err, '-ok')
plt.show()
# fig, ax = plt.subplots()
# #line, = ax.plot(x, u[0, :])
# line, = ax.plot(x, target(x, 0))

# def init():  # only required for blitting to give a clean slate.
#     line.set_ydata([np.nan] * len(x))
#     return line,


# def animate(i):
#    # line.set_ydata(u[i % 100, :])  # update the data.
#     line.set_ydata(target(x, (i % Nt) * dt ))  # update the data.
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
