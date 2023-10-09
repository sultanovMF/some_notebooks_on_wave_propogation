import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
nx = 201  # Number of grid points in x-direction
ny = 201  # Number of grid points in y-direction
dx = 1.0  # Grid spacing in x-direction
dy = 1.0  # Grid spacing in y-direction
dt = 0.001  # Time step
c = 1.0  # Wave velocity
timesteps = 500  # Number of time steps
source_pos = (100, 100)  # Source position (x, y)
source_freq = 10  # Ricker wavelet frequency

# Initialize grid
u = np.zeros((nx, ny))  # Wavefield


def ricker_wavelet(t, f):
    return (1.0 - 2.0 * (np.pi * f * t) ** 2) * np.exp(-(np.pi * f * t) ** 2)


# Main simulation loop
for i in range(timesteps):
    # Calculate Ricker wavelet source term
    source_time = i * dt
    source_term = ricker_wavelet(source_time, source_freq)
    
    # Add source term to the wavefield at the source position
    u[source_pos] += source_term
    
    # Finite difference scheme update
    u_new = np.copy(u)
    u_new[1:-1, 1:-1] = u[1:-1, 1:-1] + c**2 * dt**2 * (
        (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / dx**2 +
        (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2]) / dy**2
    )
    u = u_new


# Create animation
fig, ax = plt.subplots()
ax.set_xlim(0, nx)
ax.set_ylim(0, ny)
ax.set_xlabel('X')
ax.set_ylabel('Y')
image = ax.imshow(u, animated=True, cmap='viridis')


animation = FuncAnimation(fig, update, frames=200, interval=50, blit=True)
plt.show()
