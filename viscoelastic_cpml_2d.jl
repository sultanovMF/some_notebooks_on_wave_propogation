using Plots;
using Base.Threads;
using LoopVectorization;
using ProgressBars;


Threads.nthreads() = 8


# total number of grid points in each direction of the grid
const NX = 101
const NY = 641

# size of a grid cell
const DELTAX = 10.0
const DELTAY = DELTAX

# flags to add PML layers to the edges of the grid
const USE_PML_XMIN = true
const USE_PML_XMAX = true
const USE_PML_YMIN = true
const USE_PML_YMAX = true

# thickness of the PML layer in grid points
const NPOINTS_PML = 10

# P-velocity, S-velocity and density
const cp = 3300.0
const cs = cp / 1.732
const density = 2800.0

# total number of time steps
const NSTEP = 2000

# time step in seconds
const DELTAT = 2e-3

# parameters for the source
const f0 = 7.0
const t0 = 1.20 / f0
const factor = 1.0

# source
const ISOURCE = trunc(Int, NX - 2 * NPOINTS_PML - 1)
const JSOURCE = trunc(Int, 2 * NY / 3 + 1)

const xsource = (ISOURCE - 1) * DELTAX
const ysource = (JSOURCE - 1) * DELTAY

# angle of source force in degrees and clockwise, with respect to the vertical (Y) axis
const ANGLE_FORCE = 135.0
const DEGREES_TO_RADIANS = pi / 180
# power to compute d0 profile
NPOWER = 2

K_MAX_PML = 1 # from Stephen Gedney's unpublished class notes for class EE699, lecture 8, slide 8-11
ALPHA_MAX_PML =  2.0*pi*(f0/2.0) #  from Festa and Vilotte

# main arrays
vx = zeros(NX, NY)
vy = zeros(NX, NY)
sigma_xx = zeros(NX, NY)
sigma_yy = zeros(NX, NY)
sigma_xy = zeros(NX, NY)
lambda = zeros(NX, NY)
mu = zeros(NX, NY)
rho = zeros(NX, NY)


# arrays for the memory variables
# could declare these arrays in PML only to save a lot of memory, but proof of concept only here
memory_dvx_dx = zeros(NX, NY)
memory_dvx_dy = zeros(NX, NY)
memory_dvy_dx = zeros(NX, NY)
memory_dvy_dy = zeros(NX, NY)
memory_dsigma_xx_dx = zeros(NX, NY)
memory_dsigma_yy_dy = zeros(NX, NY)
memory_dsigma_xy_dx = zeros(NX, NY)
memory_dsigma_xy_dy = zeros(NX, NY)

# 1D arrays for the damping profiles
d_x = zeros(NX)
K_x = ones(NX)
alpha_x = zeros(NX)
a_x = zeros(NX)
b_x = zeros(NX)
d_x_half = zeros(NX)
K_x_half = ones(NX)
alpha_x_half = zeros(NX)
a_x_half = zeros(NX)
b_x_half = zeros(NX)

d_y = zeros(NY)
K_y = ones(NY)
alpha_y = zeros(NY)
a_y = zeros(NY)
b_y = zeros(NY)
d_y_half = zeros(NY)
K_y_half = ones(NY)
alpha_y_half = zeros(NY)
a_y_half = zeros(NY)
b_y_half = zeros(NY)

# thickness of the PML layer in meters
thickness_PML_x = NPOINTS_PML * DELTAX
thickness_PML_y = NPOINTS_PML * DELTAY


xoriginleft = thickness_PML_x
xoriginright = (NX-1)*DELTAX - thickness_PML_x
yoriginbottom = thickness_PML_y
yorigintop = (NY-1)*DELTAY - thickness_PML_y

Rcoef = 0.001 # reflection coefficient
d0_x = - (NPOWER + 1) * cp * log(Rcoef) / (2 * thickness_PML_x)
d0_y = - (NPOWER + 1) * cp * log(Rcoef) / (2 * thickness_PML_y)


Courant_number = cp * DELTAT * sqrt(1.0/DELTAX^2 + 1.0/DELTAY^2)
@assert Courant_number < 1 "time step is too large, simulation will be unstable", Courant_number


@fastmath @threads for i in 1:NX
  xval = DELTAX * (i - 1)
  # left edge
  if USE_PML_XMIN
    abscissa_in_PML = xoriginleft - xval
    if (abscissa_in_PML >= 0) 
      abscissa_normalized = abscissa_in_PML / thickness_PML_x
      d_x[i] = d0_x * abscissa_normalized^NPOWER

      K_x[i] = 1.0 + (K_MAX_PML - 1.0) * abscissa_normalized^NPOWER
      alpha_x[i] = ALPHA_MAX_PML * (1.0 - abscissa_normalized)
    end

    abscissa_in_PML = xoriginleft - (xval + DELTAX/2.0)
    if (abscissa_in_PML >= 0)
      abscissa_normalized = abscissa_in_PML / thickness_PML_x
      d_x_half[i] = d0_x * abscissa_normalized^NPOWER

      K_x_half[i] = 1.0 + (K_MAX_PML - 1.0) * abscissa_normalized^NPOWER
      alpha_x_half[i] = ALPHA_MAX_PML * (1.0 - abscissa_normalized)
    end
  end

  # right edge
  if (USE_PML_XMAX)
    abscissa_in_PML = xval - xoriginright
    if (abscissa_in_PML >= 0)
      abscissa_normalized = abscissa_in_PML / thickness_PML_x
      d_x[i] = d0_x * abscissa_normalized^NPOWER

      K_x[i] = 1.0 + (K_MAX_PML - 1.0) * abscissa_normalized^NPOWER
      alpha_x[i] = ALPHA_MAX_PML * (1.0 - abscissa_normalized)
    end

    abscissa_in_PML = xval + DELTAX/2.0 - xoriginright
    if (abscissa_in_PML >= 0)
      abscissa_normalized = abscissa_in_PML / thickness_PML_x
      d_x_half[i] = d0_x * abscissa_normalized^NPOWER

      K_x_half[i] = 1.0 + (K_MAX_PML - 1.0) * abscissa_normalized^NPOWER
      alpha_x_half[i] = ALPHA_MAX_PML * (1.0 - abscissa_normalized)
    end
  end

  # # just in case, for -5 at the end
  # if (alpha_x[i] < 0) 
  #   alpha_x[i] = 0
  # end
  # if (alpha_x_half[i] < 0)
  #   alpha_x_half[i] = 0
  # end

  b_x[i] = exp(- (d_x[i] / K_x[i] + alpha_x[i]) * DELTAT)
  b_x_half[i] = exp(- (d_x_half[i] / K_x_half[i] + alpha_x_half[i]) * DELTAT)

# this to avoid division by zero outside the PML
  if (abs(d_x[i]) > 1.e-6) 
    a_x[i] = d_x[i] * (b_x[i] - 1.0) / (K_x[i] * (d_x[i] + K_x[i] * alpha_x[i]))
  end

  if (abs(d_x_half[i]) > 1.e-6) 
    a_x_half[i] = d_x_half[i] * (b_x_half[i] - 1.0) / (K_x_half[i] * (d_x_half[i] + K_x_half[i] * alpha_x_half[i]))
  end
end


@fastmath @threads for j in 1:NY

# abscissa of current grid point along the damping profile
  yval = DELTAY * (j-1)

#---------- bottom edge
  if (USE_PML_YMIN) 

# define damping profile at the grid points
    abscissa_in_PML = yoriginbottom - yval
    if (abscissa_in_PML >= 0) 
      abscissa_normalized = abscissa_in_PML / thickness_PML_y
      d_y[j] = d0_y * abscissa_normalized^NPOWER

      K_y[j] = 1.0 + (K_MAX_PML - 1.0) * abscissa_normalized^NPOWER
      alpha_y[j] = ALPHA_MAX_PML * (1.0 - abscissa_normalized)
    end

# define damping profile at half the grid points
    abscissa_in_PML = yoriginbottom - (yval + DELTAY/2.0)
    if (abscissa_in_PML >= 0) 
      abscissa_normalized = abscissa_in_PML / thickness_PML_y
      d_y_half[j] = d0_y * abscissa_normalized^NPOWER

      K_y_half[j] = 10 + (K_MAX_PML - 1.0) * abscissa_normalized^NPOWER
      alpha_y_half[j] = ALPHA_MAX_PML * (1.0 - abscissa_normalized)
    end

  end

#---------- top edge
  if (USE_PML_YMAX) 

# define damping profile at the grid points
    abscissa_in_PML = yval - yorigintop
    if (abscissa_in_PML >= 0) 
      abscissa_normalized = abscissa_in_PML / thickness_PML_y
      d_y[j] = d0_y * abscissa_normalized^NPOWER

      K_y[j] = 1.0 + (K_MAX_PML - 1.0) * abscissa_normalized^NPOWER
      alpha_y[j] = ALPHA_MAX_PML * (1.0 - abscissa_normalized)
    end

# define damping profile at half the grid points
    abscissa_in_PML = yval + DELTAY/2.0 - yorigintop
    if (abscissa_in_PML >= 0) 
      abscissa_normalized = abscissa_in_PML / thickness_PML_y
      d_y_half[j] = d0_y * abscissa_normalized^NPOWER

      K_y_half[j] = 1.0 + (K_MAX_PML - 1.0) * abscissa_normalized^NPOWER
      alpha_y_half[j] = ALPHA_MAX_PML * (1.0 - abscissa_normalized)
    end

  end

  b_y[j] = exp(- (d_y[j] / K_y[j] + alpha_y[j]) * DELTAT)
  b_y_half[j] = exp(- (d_y_half[j] / K_y_half[j] + alpha_y_half[j]) * DELTAT)

# this to avoid division by zero outside the PML
  if (abs(d_y[j]) > 1.e-6)
      a_y[j] = d_y[j] * (b_y[j] - 1.0) / (K_y[j] * (d_y[j] + K_y[j] * alpha_y[j]))
  end
  if (abs(d_y_half[j]) > 1.e-6) a_y_half[j] = d_y_half[j] * (b_y_half[j] - 1.0) / (K_y_half[j] * (d_y_half[j] + K_y_half[j] * alpha_y_half[j]))
  end

end

@tturbo for j in 1:NY
  for i in 1:NX
      rho[i,j] = density
      mu[i,j] = density*cs*cs
      lambda[i,j] = density*(cp*cp - 2.0*cs*cs)
  end
end

@gif for n in ProgressBar(1:NSTEP)
  @tturbo for i in 1:NX-1
    for j in 2:NY
      lambda_half_x = 0.50 * (lambda[i+1,j] + lambda[i,j])
      mu_half_x = 0.50 * (mu[i+1,j] + mu[i,j])
      lambda_plus_two_mu_half_x = lambda_half_x + 2.0 * mu_half_x

      value_dvx_dx = (vx[i+1,j] - vx[i,j]) / DELTAX
      value_dvy_dy = (vy[i,j] - vy[i,j-1]) / DELTAY

      memory_dvx_dx[i,j] = b_x_half[i] * memory_dvx_dx[i,j] + a_x_half[i] * value_dvx_dx
      memory_dvy_dy[i,j] = b_y[j] * memory_dvy_dy[i,j] + a_y[j] * value_dvy_dy

      value_dvx_dx = value_dvx_dx / K_x_half[i] + memory_dvx_dx[i,j]
      value_dvy_dy = value_dvy_dy / K_y[j] + memory_dvy_dy[i,j]

      sigma_xx[i,j] = sigma_xx[i,j] + (lambda_plus_two_mu_half_x * value_dvx_dx + lambda_half_x * value_dvy_dy) * DELTAT

      sigma_yy[i,j] = sigma_yy[i,j] + (lambda_half_x * value_dvx_dx + lambda_plus_two_mu_half_x * value_dvy_dy) * DELTAT

    end
  end

  @tturbo for i in 2:NX
    for j in 1:NY-1
      # interpolate material parameters at the right location in the staggered grid cell
      mu_half_y = 0.5 * (mu[i,j+1] + mu[i,j])

      value_dvy_dx = (vy[i,j] - vy[i-1,j]) / DELTAX
      value_dvx_dy = (vx[i,j+1] - vx[i,j]) / DELTAY

      memory_dvy_dx[i,j] = b_x[i] * memory_dvy_dx[i,j] + a_x[i] * value_dvy_dx
      memory_dvx_dy[i,j] = b_y_half[j] * memory_dvx_dy[i,j] + a_y_half[j] * value_dvx_dy

      value_dvy_dx = value_dvy_dx / K_x[i] + memory_dvy_dx[i,j]
      value_dvx_dy = value_dvx_dy / K_y_half[j] + memory_dvx_dy[i,j]

      sigma_xy[i,j] = sigma_xy[i,j] + mu_half_y * (value_dvy_dx + value_dvx_dy) * DELTAT
    end
  end

  # compute velocity and update memory variables for C-PML
  @tturbo for i in 2:NX
    for j in 2:NY
      value_dsigma_xx_dx = (sigma_xx[i,j] - sigma_xx[i-1,j]) / DELTAX
      value_dsigma_xy_dy = (sigma_xy[i,j] - sigma_xy[i,j-1]) / DELTAY

      memory_dsigma_xx_dx[i,j] = b_x[i] * memory_dsigma_xx_dx[i,j] + a_x[i] * value_dsigma_xx_dx
      memory_dsigma_xy_dy[i,j] = b_y[j] * memory_dsigma_xy_dy[i,j] + a_y[j] * value_dsigma_xy_dy

      value_dsigma_xx_dx = value_dsigma_xx_dx / K_x[i] + memory_dsigma_xx_dx[i,j]
      value_dsigma_xy_dy = value_dsigma_xy_dy / K_y[j] + memory_dsigma_xy_dy[i,j]

      vx[i,j] = vx[i,j] + (value_dsigma_xx_dx + value_dsigma_xy_dy) * DELTAT / rho[i,j]
    end
  end

  @tturbo for i in 1:NX-1
    for j in 1:NY-1
      rho_half_x_half_y = 0.25 * (rho[i,j] + rho[i+1,j] + rho[i+1,j+1] + rho[i,j+1])

      value_dsigma_xy_dx = (sigma_xy[i+1,j] - sigma_xy[i,j]) / DELTAX
      value_dsigma_yy_dy = (sigma_yy[i,j+1] - sigma_yy[i,j]) / DELTAY

      memory_dsigma_xy_dx[i,j] = b_x_half[i] * memory_dsigma_xy_dx[i,j] + a_x_half[i] * value_dsigma_xy_dx
      memory_dsigma_yy_dy[i,j] = b_y_half[j] * memory_dsigma_yy_dy[i,j] + a_y_half[j] * value_dsigma_yy_dy

      value_dsigma_xy_dx = value_dsigma_xy_dx / K_x_half[i] + memory_dsigma_xy_dx[i,j]
      value_dsigma_yy_dy = value_dsigma_yy_dy / K_y_half[j] + memory_dsigma_yy_dy[i,j]

      vy[i,j] = vy[i,j] + (value_dsigma_xy_dx + value_dsigma_yy_dy) * DELTAT / rho_half_x_half_y
    end
  end

  a = pi*pi*f0*f0
  t = (n-1)*DELTAT
  source_term = - factor * 2.0*a*(t-t0)*exp(-a*(t-t0)^2)

  force_x = sin(ANGLE_FORCE * DEGREES_TO_RADIANS) * source_term
  force_y = cos(ANGLE_FORCE * DEGREES_TO_RADIANS) * source_term

  rho_half_x_half_y = 0.25 * (rho[ISOURCE,JSOURCE] + rho[ISOURCE + 1,JSOURCE] + rho[ISOURCE + 1,JSOURCE + 1] + rho[ISOURCE,JSOURCE + 1])

  vx[ISOURCE,JSOURCE] = vx[ISOURCE,JSOURCE] + force_x * DELTAT / rho[ISOURCE,JSOURCE]
  vy[ISOURCE,JSOURCE] = vy[ISOURCE,JSOURCE] + force_y * DELTAT / rho_half_x_half_y

  heatmap(vx[:, :],framestyle = :box, aspect_ratio = :equal, xlabel = "X", ylabel = "Y", title = "Wave Propagation", color = :seismic, size = (900, 900))
end every 10