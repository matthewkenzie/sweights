import os
import sys
sys.path.append("/Users/matt/Scratch/stats/SWeights")
from utils import hist
import numpy as np
import matplotlib.pyplot as plt

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('-s','--sfile', help='Signal toy file')
parser.add_argument('-b','--bfile', help='Background toy file')
parser.add_argument('-o','--outf' , default=None, help='Save plot to this file')
parser.add_argument('-i','--interactive', default=False,action='store_true',help='Show interactive plots at the end')
parser.add_argument('-e','--eff', default=False, action='store_true', help='Plot efficiency as well')
opts = parser.parse_args()

def read_opts_from_file_name(fname):
  assert('/' in fname)
  dirs = fname.split('/')
  modname = dirs[-2]
  assert( modname in ['sigmodel','sigweffmodel','bkgmodel','bkgnf','bkgweffmodel','bkgnfweffmodel'] )
  return modname

# ranges
mrange = (5000,5600)
trange = (0,10)

# bins (for plots)
mbins = 50
tbins = 50

# spars
mmu = 5280
msg = 30
tmu = -1

# bpars
mlb = 400
tsg = 2
tlb = 4
slb = 300
smu = 0.3
ssg = 0.8

# effpars
ea  = 0.2
eb  = 0.2
ed  = -0.15

from models import effmodel, sigmodel, sigweffmodel, bkgmodel, bkgnfmodel, bkgweffmodel, bkgnfweffmodel

assert( os.path.exists(opts.sfile) )
assert( os.path.exists(opts.bfile) )

sname = read_opts_from_file_name(opts.sfile)
bname = read_opts_from_file_name(opts.bfile)

smod = None
bmod = None

if   sname=='sigmodel':
  smod = sigmodel(mrange,trange,mmu,msg,tlb,cache='load')

elif sname=='sigweffmodel':
  smod = sigweffmodel(mrange,trange,mmu,msg,tlb,ea,eb,ed,cache='load')

else:
  raise RuntimeError('Not a valid model option', sname)

if bname=='bkgmodel':
  bmod = bkgmodel(mrange,trange,mlb,tmu,tsg,cache='load')

elif bname=='bkgweffmodel':
  bmod = bkgweffmodel(mrange,trange,mlb,tmu,tsg,ea,eb,ed,cache='load')

elif bname=='bkgnfmodel':
  bmod = bkgnfmodel(mrange,trange,mlb,tmu,tsg,slb,smu,ssg,cache='load')

elif bname=='bkgnfweffmodel':
  bmod = bkgnfweffmodel(mrange,trange,mlb,tmu,tsg,slb,smu,ssg,ea,eb,ed,cache='load')

else:
  raise RuntimeError('Not a valid model option', bname)

bkg_toy = np.load( opts.bfile )
sig_toy = np.load( opts.sfile )
nbkg = len(bkg_toy)
nsig = len(sig_toy)

toy_vals = np.concatenate((bkg_toy,sig_toy))

fig, ax = plt.subplots(1,2,figsize=(16,6))

mhc, mhw = hist(toy_vals[:,0], range=mrange, bins=mbins)
thc, thw = hist(toy_vals[:,1], range=trange, bins=tbins)

m  = np.linspace(*mrange,200)
mBN = nbkg*(mrange[1]-mrange[0])/mbins
mSN = nsig*(mrange[1]-mrange[0])/mbins
ms  = mSN*smod.pdfm(m)
mb  = mBN*bmod.pdfm(m)
if opts.eff:
  mse = smod.effm(m)
  mse = mse*np.max(ms+mb)/np.max(mse)
  mbe = bmod.effm(m)
  mbe = mbe*np.max(ms+mb)/np.max(mbe)


t = np.linspace(*trange,200)
tBN = nbkg*(trange[1]-trange[0])/tbins
tSN = nsig*(trange[1]-trange[0])/tbins
ts = tSN*smod.pdft(t)
tb = tBN*bmod.pdft(t)
if opts.eff:
  tse = smod.efft(t)
  tse = tse*np.max(ts+tb)/np.max(tse)
  tbe = bmod.efft(t)
  tbe = tbe*np.max(ts+tb)/np.max(tbe)

ax[0].errorbar( mhc, mhw, mhw**0.5, fmt='ko', label='Toy Data' )
ax[0].plot(m, ms   , 'g--' , label='True S pdf' )
ax[0].plot(m, mb   , 'r--' , label='True B pdf')
ax[0].plot(m, ms+mb, 'b-'  , label='True S+B pdf')
if opts.eff:
  ax[0].plot(m, mse  , c='0.75', label='S efficiency')
  ax[0].plot(m, mbe  , 'k:'  , label='B efficiency')

ax[0].set_xlabel('Mass [GeV]')
ax[0].set_ylabel('Events')
ax[0].legend()

ax[1].errorbar( thc, thw, thw**0.5, fmt='ko', label='Toy Data' )
ax[1].plot(t, ts   , 'g--', label='True S pdf')
ax[1].plot(t, tb   , 'r--', label='True B pdf')
ax[1].plot(t, ts+tb, 'b-' , label='True S+B pdf')
if opts.eff:
  ax[1].plot(t, tse  , c='0.75', label='S efficiency')
  ax[1].plot(t, tbe  , 'k:'  , label='B efficiency')

ax[1].set_xlabel('Time [ps]')
ax[1].set_ylabel('Events')
ax[1].legend()

fig.tight_layout()
if opts.outf is not None: fig.savefig(opts.outf.replace('.pdf','')+'1d.pdf')

# 2D plots
fig, ax = plt.subplots(1+opts.eff,2,figsize=(16,6*(opts.eff+1)),squeeze=False)
x,y = np.meshgrid(m,t)

ax[0,0].set_title('True Background PDF')
cb1 = ax[0,0].contourf(x,y,bmod.pdf(x,y))
fig.colorbar(cb1,ax=ax[0,0])
ax[0,0].set_xlabel('Mass [GeV]')
ax[0,0].set_ylabel('Time [ps]')

ax[0,1].set_title('True Signal PDF')
cb2 = ax[0,1].contourf(x,y,smod.pdf(x,y))
fig.colorbar(cb2,ax=ax[0,1])
ax[0,1].set_xlabel('Mass [GeV]')
ax[0,1].set_ylabel('Time [ps]')

if opts.eff:
  ax[1,0].set_title('Background Efficiency')
  cb3 = ax[1,0].contourf(x,y,bmod.effmt(x,y))
  fig.colorbar(cb3,ax=ax[1,0]).set_label('Efficiency')
  ax[1,0].set_xlabel('Mass [GeV]')
  ax[1,0].set_ylabel('Time [ps]')

  ax[1,1].set_title('Signal Efficiency')
  cb4 = ax[1,1].contourf(x,y,smod.effmt(x,y))
  fig.colorbar(cb4,ax=ax[1,1]).set_label('Efficiency')
  ax[1,1].set_xlabel('Mass [GeV]')
  ax[1,1].set_ylabel('Time [ps]')

fig.tight_layout()
if opts.outf is not None: fig.savefig(opts.outf.replace('.pdf','')+'2d.pdf')

if opts.interactive: plt.show()
