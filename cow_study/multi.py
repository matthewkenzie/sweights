from scipy.stats import rv_continuous
from scipy.stats import expon, norm

class bkglt_gen(rv_continuous):
  def pdf(self, x, mm, lb, mu, sg):
    exp = expon(mm,lb)
    nrm = norm(mu,sg)
    #pr = np.prod(x,axis=-1)
    return exp.pdf(x[...,0]) * nrm.pdf(x[...,1])

bkglt = bkglt_gen(name='bkglt')

## tests

from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(5000,5600,50)
y = np.linspace(0,10,50)
X, Y = np.meshgrid(x,y)
pos = np.dstack((X,Y))
corr = -0.5
sx = 600
sy = 0.5
g = mvn( [4000,0], [[sx**2,corr*sx*sy],[corr*sx*sy,sy**2]] )
gn = np.diff(g.cdf( [ [5000,0],[5600,10] ] ))
#b = bkglt( 5000, 400, 0, 0.5 )

gz = g.pdf(pos)

def f(t,m):
  return g.pdf( [m,t] ) / gn

from scipy.integrate import nquad, dblquad
print(nquad( f, ranges=((5000,5600),(0,3)) ))
print(dblquad( f, 5000, 5600, lambda x: 0, lambda x: 10 ) )
#gb = b.pdf(pos)

#print(gz.shape, gb.shape)

exp = expon(5000,400)

fig, ax = plt.subplots(2,2,figsize=(12,8))
ax[0,0].contourf( X, Y, gz )
#ax[0,1].contourf( X, Y, gb )
ax[1,0].plot( x, g.pdf(np.stack((x,np.full(x.shape,2)),axis=1))/g.pdf([5000,2]), label='$t=2$' )
ax[1,0].plot( x, g.pdf(np.stack((x,np.full(x.shape,4)),axis=1))/g.pdf([5000,4]), label='$t=4$' )
ax[1,0].plot( x, g.pdf(np.stack((x,np.full(x.shape,6)),axis=1))/g.pdf([5000,6]), label='$t=6$' )
ax[0,1].plot( y, g.pdf(np.stack((np.full(y.shape,5100),y),axis=1))/g.pdf([5100,0]), label='$m=5100$' )
ax[0,1].plot( y, g.pdf(np.stack((np.full(y.shape,5300),y),axis=1))/g.pdf([5300,0]), label='$m=5300$' )
ax[0,1].plot( y, g.pdf(np.stack((np.full(y.shape,5500),y),axis=1))/g.pdf([5500,0]), label='$m=5500$' )
ax[0,1].set_yscale('log')
#ax[1,0].plot( x, b.pdf( np.stack((x,np.full(x.shape,2)),axis=1) ), label='$t=2$' )
#ax[1,0].plot( x, b.pdf( np.stack((x,np.full(x.shape,4)),axis=1) ), label='$t=4$' )
#ax[1,0].plot( x, b.pdf( np.stack((x,np.full(x.shape,6)),axis=1) ), label='$t=6$' )
#ax[1,0].plot( x, b.pdf( np.stack((x,np.full(x.shape,8)),axis=1) ), label='$t=8$' )
#ax[1,1].plot( y, b.pdf( np.stack((np.full(y.shape,5100),y),axis=1) ) )
plt.show()
