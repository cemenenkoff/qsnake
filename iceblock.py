#0.917 g/cm³ at 0 °C
rho = 0.917*(1/3.5314710**(-5))*(1/10**3)
vol = 5*5*5 #ft^3

mass = rho*vol #kg/ft^3*ft^3
ydist = 30*(0.3048/1)

import numpy as np

def v(v0,a,r,r0):
    return np.sqrt(v0**2+2*a*(r-r0))

print(v(0,-9.81,0,ydist))

print(1)