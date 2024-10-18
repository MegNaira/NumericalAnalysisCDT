import numpy as np
import matplotlib.pyplot as plt

u = 1.0
nx = 80
x = np.linspace(0.0, 1.0, nx + 1)
nt = 40
dx = 2./nx
dt = 1./nt

# c = u*dt/dx
# If c > 1, the solution is unstable. If c < 1, the solution is stable and not damping.

# The spacial resolution
# The time step
# The initial conditions and arrays for the old and new time steps
phi = np.where(x%1. < 0.5, np.power(np.sin(2*x*np.pi), 2), 0.)
phiOld = phi.copy()
def analytic(x,t):
    return np.where((x-u*t)%1. < 0.5, np.power(np.sin(2*np.pi*(x-u*t)),2), 0.)


# Plot the initial conditions
plt.plot(x, phi, 'k', label = 'initial_conditions')
plt.legend(loc = 'best')
plt.ylabel('$\phi$')
plt.axhline(0, linestyle = ':', color = 'black')
plt.ylim([0,1])

plt. pause(1)
# Loop over all time steps
for n in range(nt):
  for j in range(1,nx+1):
# (avoiding boundary conditions)
    phi[j] = phiOld[j] - u*dt*(phiOld[j] - phiOld[j - 1])/dx
  phi[0]=phi[-1]  
  phiOld = phi.copy()
  if int(n)%(nt/10)==0: 
    plt.cla()
    plt.plot(x,analytic(x,n*dt),'r',label='analytic')
    plt.plot(x, phi, 'b', label = 'Time ' + str(n*dt))
    plt.legend(loc = 'best')
    plt.ylabel('$\phi$')
    plt.ylim([0,1])
    plt.pause(0.05)
