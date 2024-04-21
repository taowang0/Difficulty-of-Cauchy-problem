""" 
This file was built to solve numerically 1D Burgers' equation wave equation with the FFT. The equation corresponds to :

$\dfrac{\partial u}{\partial t} + \mu u\dfrac{\partial u}{\partial x} = \nu \dfrac{\partial^2 u}{\partial x^2}$
 
where
 - u represent the signal
 - x represent the position
 - t represent the time
 - nu and mu are constants to balance the non-linear and diffusion terms.

Copyright - © SACHA BINDER - 2021
"""

############## MODULES IMPORTATION ###############
import numpy as np
from scipy.integrate import odeint
import viz_tools


############## SET-UP THE PROBLEM ###############

mu = -1
nu = 0.001 #kinematic viscosity coefficient
    
#Spatial mesh
L_x = 1 #Range of the domain according to x [m]
dx = 0.001 # 0.001 #Infinitesimal distance
# N_x = int(L_x/dx) #Points number of the spatial mesh
# X = np.linspace(0,L_x,N_x) #Spatial array
N_x = int(L_x/dx) * 2 + 1
X = np.linspace(-L_x,L_x,N_x) #Spatial array

#Temporal mesh
L_t = 1 #Duration of simulation [s]
dt = 0.01 # 0.001  #Infinitesimal time
N_t = int(L_t/dt) #Points number of the temporal mesh
T = np.linspace(0,L_t,N_t) #Temporal array

#Wave number discretization
k = 2*np.pi*np.fft.fftfreq(N_x, d = dx)


#Def of the initial condition    
# u0 = np.exp(-(X-3)**2/2) #Single space variable fonction that represent the wave form at t = 0
u0 = np.sin(np.pi * (X / 2))
# viz_tools.plot_a_frame_1D(X,u0,0,L_x,0,1.2,'Initial condition')

############## EQUATION SOLVING ###############

#Definition of ODE system (PDE ---(FFT)---> ODE system)
def burg_system(u,t,k,mu,nu):
    #Spatial derivative in the Fourier domain
    u_hat = np.fft.fft(u)
    u_hat_x = 1j*k*u_hat
    u_hat_xx = -k**2*u_hat
    
    #Switching in the spatial domain
    u_x = np.fft.ifft(u_hat_x)
    u_xx = np.fft.ifft(u_hat_xx)
    
    #ODE resolution
    u_t = -mu*u*u_x + nu*u_xx
    return u_t.real
    

#PDE resolution (ODE system resolution)
U = odeint(burg_system, u0, T, args=(k,mu,nu,), mxstep=5000).T
np.save('data/U.npy', U)
np.save('data/X.npy', X)
np.save('data/T.npy', T)
# U = np.load('U.npy')

print(U.shape)


############## PLOT ###############

#Anim
# anim = viz_tools.anim_1D(X,U,dt,2, True, (0,L_x), (0,1.2))

#Plots
viz_tools.plot_spatio_temp_3D(X,T,U, 'figures/Burgers_solution.pdf')
# viz_tools.plot_spatio_temp_flat(X,U.T,T)
# viz_tools.plot_sequence(X,U.T)

