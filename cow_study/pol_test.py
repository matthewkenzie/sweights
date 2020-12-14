import numpy as np
from scipy.integrate import quad

xr = (5000,5600)

x = np.linspace(*xr,100)

fold = lambda m: (m-xr[0])/(xr[1]-xr[0])

pn  = lambda n, m: (n+1)*m**n / ( xr[1]**(n+1) - xr[0]**(n+1) )
p0  = lambda m: fold(m)**0 / np.diff(xr)
p1  = lambda m: 2*fold(m) / np.diff(xr)
p2  = lambda m: 3*fold(m)**2 / np.diff(xr)
p3  = lambda m: 4*fold(m)**3 / np.diff(xr)
p4  = lambda m: 5*fold(m)**4 / np.diff(xr)
p5  = lambda m: 6*fold(m)**5 / np.diff(xr)

pols = [p0,p1,p2,p3,p4,p5]

for p in pols:
  print( quad(p, *xr) )

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

fig, ax = plt.subplots(2,2, figsize=(12,8))

l, = ax[0].plot( x, 600*p0(x) )
ax[0].set_ylim(0,6)

axamp = plt.axes([0.25,0.03,0.5,0.02])

samp = Slider(axamp,'nTerms',0,5, valstep=1,valinit=0)

def update(val):
  l.set_ydata( 600*pols[samp.val](x) )
  #fig.canvas
samp.on_changed(update)

#plt.plot( x, p0(x) )
#plt.plot( x, p1(x) )
#plt.plot( x, p2(x) )
#plt.plot( x, p3(x) )
#plt.plot( x, p4(x) )
#plt.plot( x, p5(x) )
#plt.ylim = (0,6)
plt.show()
