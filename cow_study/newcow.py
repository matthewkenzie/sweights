import numpy as np
import numba as nb
import math

kwd = {"parallel": True, "fastmath": True}

# custom faster computation of pdf and cdf of a normal distribution
@nb.njit(**kwd)
def norm_pdf(x, mu, sigma):
    invs = 1.0 / sigma
    z = (x - mu) * invs
    invnorm = 1 / np.sqrt(2 * np.pi) * invs
    return np.exp(-0.5 * z ** 2) * invnorm

@nb.njit(**kwd)
def nb_erf(x):
    y = np.empty_like(x)
    for i in nb.prange(x.shape[0]):
        y[i] = math.erf(x[i])
    return y

@nb.njit(**kwd)
def norm_cdf(x, mu, sigma):
    invs = 1.0 / (sigma * np.sqrt(2))
    z = (x - mu) * invs
    return 0.5 * (1 + nb_erf(z))

#@nb.njit(**kwd)
def norm_int(xmin, xmax, mu, sigma):
    c = norm_cdf(np.array([xmin,xmax]), mu, sigma)
    return c[1] - c[0]

#print(norm_cdf(np.atleast_1d(1),0,1))

import matplotlib.pyplot as plt

mrange = (5000,5600)
mmu = 5280
msg = 30
mlb = 400
trange = (0,10)
tmu = -1
tsg = 2
tlb = 4

from scipy.stats import truncnorm, truncexpon
from scipy.integrate import quad, nquad

def mynorm(xmin,xmax,mu,sg):
  a, b = (xmin-mu)/sg, (xmax-mu)/sg
  return truncnorm(a,b,mu,sg)

def myexp(xmin,xmax,lb):
  return truncexpon( (xmax-xmin)/lb, xmin, lb )

def mybkg(m,t,mrange,trange,lb,mu,sg, slb, smu, ssg, mproj=False, tproj=False):
  # calc lambda for mass part
  dt = 2 * (t - trange[0]) / (trange[1] - trange[0]) - 1
  flb = lb + slb*dt
  # calc mu, sigma for time part
  dm = 2 * (m - mrange[0]) / (mrange[1] - mrange[0]) - 1
  fmu = mu + smu*dm
  fsg = sg + ssg*dm

  mpdf = myexp(*mrange, flb)
  tpdf = mynorm(*trange, fmu, fsg)

  if mproj: return mpdf.pdf(m)
  if tproj: return tpdf.pdf(t)
  return mpdf.pdf(m)*tpdf.pdf(t)

def bkgN(m,t):
  return mybkg(m,t,mrange,trange,mlb,tmu,tsg,300,0.2,0.8)

N, Nerr = nquad( bkgN, (mrange,trange) )
#print(N,Nerr)

def bkgpdf(m,t, mproj=False, tproj=False):
  return mybkg(m,t,mrange,trange, mlb, tmu, tsg, 300, 0.2, 0.8, mproj=mproj, tproj=tproj) / N

def bkgpdfm(m):
  f = lambda t: mybkg(m,t,mrange,trange,mlb,tmu,tsg,300,0.2,0.8)
  return quad(f,*trange)[0]

def bkgpdft(t):
  f = lambda m: mybkg(m,t,mrange,trange,mlb,tmu,tsg,300,0.2,0.8)
  return quad(f,*mrange)[0]

# plot pdf projs
fig, ax = plt.subplots(1,2,figsize=(16,6))

# pdfs
smpdf = mynorm(*mrange,mmu, msg)
stpdf = myexp(*trange,tlb)
bpdf = bkgpdf
bpdfm = np.vectorize(bkgpdfm)
bpdft = np.vectorize(bkgpdft)

from tqdm import tqdm
# for generation can be find maximum of bkgpdf
#from iminuit import Minuit
#f = lambda m,t: -bpdf(m,t)
#m = Minuit(f,m=5200,t=1,limit_m=mrange,limit_t=trange,pedantic=False)
#m.migrad()
#print(m.params)
#print(m.fval)
#input()
# some background generation
bkg_majorant = bpdf(mrange[0],trange[0])
nevs = 100
ngen = 0
toy_vals = []
bar = tqdm(total=nevs)
while ngen < nevs:
  accept = False
  m = None
  t = None
  while not accept:
    m = np.random.uniform(*mrange)
    t = np.random.uniform(*trange)
    p = bpdf(m,t)
    h = np.random.uniform(0,bkg_majorant)
    if h<p: accept=True

  toy_vals.append( (m,t) )
  bar.update()
  ngen += 1

print(toy_vals)
toy_vals = np.array(toy_vals)
# m plot
m = np.linspace(*mrange,200)
#x, we = np.histogram(toy_vals[:,0], range=mrange,bins=10)
ax[0].hist(toy_vals[:,0], range=mrange, bins=10)
ax[0].plot(m, smpdf.pdf(m), label='S pdf' )
for tval in np.linspace(*trange,3):
  ax[0].plot(m, bpdf(m,tval,mproj=True), label=f'B t={tval}' )
#ax[0].plot(m, bpdfm(m), label='B pdf')
ax[0].legend()

# t plot
t = np.linspace(*trange,200)
ax[1].plot(t, stpdf.pdf(t), label='S pdf' )
for mval in np.linspace(*mrange,3):
  ax[1].plot(t, bpdf(mval,t,tproj=True), label=f'B m={mval}')
#ax[1].plot(t, bpdft(t), label='B pdf')
ax[1].legend()
fig.tight_layout()

# 2D lad
fig, ax = plt.subplots(1,1,figsize=(8,6))
x,y = np.meshgrid(m,t)
ax.contourf(x,y,bpdf(x,y))

print( nquad( bpdf, (mrange,trange) ) )


plt.show()
