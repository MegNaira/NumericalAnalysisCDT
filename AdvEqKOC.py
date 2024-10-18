import numpy as np
import matplotlib.pyplot as plt

u = 1.0
nx = 40
x = np.linspace(0.0, 1.0, nx + 1)
nt = 80
dx = 1./nx
dt = 1./nt

c = u*dt/dx
print('Courant number =', c)
# FTBS is stable and damping for 0<=c<=1

# The initial conditions and arrays for the old arrays for the old and new time steps
def analytic(x,t):
    return np.where((x-u*t)%1. < 0.5, np.power(np.sin(2*np.pi*(x-u*t)),2), 0.)
phi=np.array(analytic(x,0))
phiOld=phi.copy()

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
    phi[j] = phiOld[j] - c*(phiOld[j] - phiOld[j - 1])
  phi[0]=phi[-1]  
  phiOld = phi.copy()
  if n%5==0: 
    plt.cla()
    plt.plot(x,analytic(x,(n+1)*dt),'r',label='analytic')
    plt.plot(x, phi, 'b', label = 'Time ' + str((n+1)*dt))
    plt.legend(loc = 'best')
    plt.ylabel('$\phi$')
    plt.ylim([0,1])
    plt.pause(0.05)
