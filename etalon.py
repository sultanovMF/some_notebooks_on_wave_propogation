import numpy as np
from numpy import pi
from numba import jit 
import matplotlib.pyplot as plt
from scipy import sparse

import time
import datetime


# solving Ax = Y
# a = A.diagonal(-1)
# b = A.diagonal()
# c = A.diagonal(1)
# thus if A is of n by n, then a and c are of (n-1) by 1

#   Ax = Y
#   [b(1)   c(1)               ] [ x(1) ]   [ Y(1) ]
#   [a(1)   b(2)   c(2)        ] [ x(2) ]   [ Y(2) ]
#   [    ...   ...   ...       ] [ ...  ] = [ ...  ]
#   [      a(n)   b(n-1) c(n-1)] [x(n-1)]   [Y(n-1)]
#   [            a(n-1)   b(n) ] [ x(n) ]   [ Y(n) ]
@jit(nopython=True,parallel=True,fastmath = True)
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




stime = time.time()

h = 1
tau = h/10000
para = 3/4 # parameter

Nx = para*1600/h - 1
Ny = para*1600/h - 1
Nz = para*1800/h - 1

Nx = int(Nx)
Ny = int(Ny)
Nz = int(Nz)

print(Nx, Ny, Nz)
center = para*800/h - 1
center = int(center)

mediachange = para*1173
mediachange = int(mediachange)

TT = 0.1
T = TT*para*0.4/tau # t(T) = TT*0.3 s
T = int(T)

snapshots = np.zeros((Nx,Nz,T))


fp = 10
dr = 0.5/fp

c2 = 1200**2*np.ones((Nx,Ny,Nz))
v = np.zeros((Nx,Ny,Nz))
w = np.zeros((Nx,Ny,Nz))

# v[center,center,center] = 1/2*tau**2*(1-2*pi**2*fp**2*((1-1)*tau-dr)**2)*np.exp(-pi**2*fp**2*((1-1)*tau-dr)**2)
v[center,center,center] = 1/2*tau**2*(1-2*pi**2*fp**2*((1-1)*tau-dr)**2)*np.exp(-pi**2*fp**2*((1-1)*tau-dr)**2) \
                        - 1/6*tau**3*(-4*pi**2*fp**2*(-dr)*np.exp(-pi**2*fp**2*((1-1)*tau-dr)**2)-(1-2*pi**2*fp**2 \
                        *dr**2)*pi**2*fp**2*2*(-dr)*np.exp(-pi**2*fp**2*((1-1)*tau-dr)**2))
for k in range(mediachange//h,Nz):
  c2[:,:,k] = 2500**2*np.ones((Nx,Ny))


a0 = 1
a1 = 1/10
b0 = -12/(5*h**2)
b1 = 6/(5*h**2)

# Ax = a0*np.eye(Nx)+a1*np.eye(Nx,k=1)+a1*np.eye(Nx,k=-1)
Bx = b0*np.eye(Nx)+b1*np.eye(Nx,k=1)+b1*np.eye(Nx,k=-1)

# Ay = a0*np.eye(Ny)+a1*np.eye(Ny,k=1)+a1*np.eye(Ny,k=-1)
By = b0*np.eye(Ny)+b1*np.eye(Ny,k=1)+b1*np.eye(Ny,k=-1)

# Az = a0*np.eye(Nz)+a1*np.eye(Nz,k=1)+a1*np.eye(Nz,k=-1)
Bz = b0*np.eye(Nz)+b1*np.eye(Nz,k=1)+b1*np.eye(Nz,k=-1)

Bx = sparse.csr_matrix(Bx)
By = sparse.csr_matrix(By)
Bz = sparse.csr_matrix(Bz)

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

for nn in range(0,T):
  
  if nn % 10 == 0:
    now = datetime.datetime.now()
    print('Step ',nn,'/',T,' @ ',now.strftime("%H:%M:%S"))
  
  for j in range(0,Ny):
    for k in range(0,Nz):
      u_xx[:,j,k] = TDMA(a1x,a0x,a1x,Bx @ w[:,j,k])
  
  for i in range(0,Nx):
    for k in range(0,Nz):
      u_yy[i,:,k] = TDMA(a1y,a0y,a1y,By @ w[i,:,k])
  
  for i in range(0,Nx):
    for j in range(0,Ny):
      u_zz[i,j,:] = TDMA(a1z,a0z,a1z,Bz @ w[i,j,:])
  
  f[center,center,center] = (1-2*pi**2*fp**2*(nn*tau-dr)**2)*np.exp(-pi**2*fp**2*(nn*tau-dr)**2)

  u = tau**2*(c2*(u_xx + u_yy + u_zz) + f) + 2*w - v
  v = w
  w = u
	

  snapshots[:,:,nn] = u[:,center,:]



# animation
from IPython.display import HTML
import matplotlib.animation as animation

snapshots_trans = np.transpose(snapshots,(1,0,2))

fig, ax = plt.subplots()

extent = (0,1200,1350,0)
plt.xticks([0, 400, 800, 1200])
plt.yticks([0, 450, 900, 1350])
plt.xlabel('x',fontsize = 14)
plt.ylabel('z',fontsize = 14)
plt.plot([0, 1200],[880, 880],'r',lw = 1)

ims = []
for i in range(0,T):
  im = plt.imshow(snapshots_trans[:,:,i],interpolation ='lanczos' ,cmap = 'summer',extent=extent)
  plt.clim(-10**-8,10**-8)
  t = 0.0025*i
  title = ax.text(600,-50,'t = %1.2f s' %t,size=plt.rcParams["axes.titlesize"],ha="center")
  ims.append([im, title])

ani = animation.ArtistAnimation(fig, ims, interval=30,blit=False)
plt.colorbar()
plt.show()
