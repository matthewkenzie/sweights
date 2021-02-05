import os
import sys
sys.path.append("/Users/matt/Scratch/stats/SWeights")
from utils import hist
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

model_choices = list(range(1,7)) + ['sig','sigeff','bkg','bkgeff','bkgnf','bkgnfeff']

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('-m','--model'   , type=int, choices=model_choices  , metavar='MODEL',help='Generating model: \n'
                                                                               '  1: sig \n'
                                                                               '  2: sigeff \n'
                                                                               '  3: bkg \n'
                                                                               '  4: bkgeff \n'
                                                                               '  5: bkgnf \n'
                                                                               '  6: bkgnfeff' )
parser.add_argument('-n','--ntoys'   , type=int, default=1               , help='Number of toys to run')
parser.add_argument('-j','--startn'  , type=int, default=1               , help='Toy number to start on')
parser.add_argument('-e','--nevs'    , type=int, default=1000            , help='Number of events per toy')
parser.add_argument('-p','--poiss'   , default=False, action='store_true', help='Poisson flucutate number of events in toys')
parser.add_argument('-o','--outdir'  , default='toys'                    , help='Save location')
parser.add_argument('-O','--overwrite', default=False,action='store_true', help='Overwite already existing toys')
parser.add_argument('-P','--plot'    , default=False, action='store_true', help='Make plots')
parser.add_argument('-i','--interactive', default=False,action='store_true',help='Show interactive plots at the end')
opts = parser.parse_args()

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

mod = None
if   opts.model == 1 or opts.model=='sig':
  mod = sigmodel(mrange,trange,mmu,msg,tlb,cache='load')

elif opts.model == 2 or opts.model=='sigeff':
  mod = sigweffmodel(mrange,trange,mmu,msg,tlb,ea,eb,ed,cache='load')

elif opts.model == 3 or opts.model=='bkg':
  mod = bkgmodel(mrange,trange,mlb,tmu,tsg,cache='load')

elif opts.model == 4 or opts.model=='bkgeff':
  mod = bkgweffmodel(mrange,trange,mlb,tmu,tsg,ea,eb,ed,cache='load')

elif opts.model == 5 or opts.model=='bkgnf':
  mod = bkgnfmodel(mrange,trange,mlb,tmu,tsg,slb,smu,ssg,cache='load')

elif opts.model == 6 or opts.model=='bkgnfeff':
  mod = bkgnfweffmodel(mrange,trange,mlb,tmu,tsg,slb,smu,ssg,ea,eb,ed,cache='load')

else:
  raise RuntimeError('Not a valid model option', opts.model)

# make output dir
os.system('mkdir -p %s/%s'%(opts.outdir, mod.name) )
if opts.plot:
  os.system('mkdir -p plots/%s'%(mod.name) )

pbar = tqdm(desc='Generating toys', total=opts.ntoys)

for it in range(opts.startn, opts.startn+opts.ntoys):

  size = np.random.poisson(opts.nevs) if opts.poiss else opts.nevs

  fname = 'toy_n%d'%opts.nevs
  if opts.poiss: fname += '_poiss'
  fname += '_t%d'%it
  fname += '.npy'

  outf = os.path.join(opts.outdir, mod.name, fname)

  if os.path.exists(outf) and not opts.overwrite:
    raise RuntimeError('Toy file', outf, 'already exists and overwrite is off')

  mod.generate(size=size,save=outf,progress=False,seed=model_choices.index(opts.model)*10000+it)
  pbar.update()

  # plot if asked
  if opts.plot:

    fig, ax = plt.subplots(1,2,figsize=(16,6))

    toy_vals = np.load(outf)

    mhc, mhw = hist(toy_vals[:,0], range=mrange, bins=mbins)
    thc, thw = hist(toy_vals[:,1], range=trange, bins=tbins)

    m  = np.linspace(*mrange,200)
    mN = len(toy_vals)*(mrange[1]-mrange[0])/mbins

    t = np.linspace(*trange,200)
    tN = len(toy_vals)*(trange[1]-trange[0])/tbins

    ax[0].errorbar( mhc, mhw, mhw**0.5, fmt='ko', label='Toy Data')
    ax[0].plot( m, mN*mod.pdfm(m), 'b-', label='PDF')
    ax[0].set_xlabel('Mass [GeV]')
    ax[0].set_ylabel('Events')
    ax[0].legend()

    ax[1].errorbar( thc, thw, thw**0.5, fmt='ko', label='Toy Data' )
    ax[1].plot( t, tN*mod.pdft(t), 'b-', label='PDF')
    ax[1].set_xlabel('Time [ps]')
    ax[1].set_ylabel('Events')
    ax[1].legend()

    fig.tight_layout()
    fig.savefig( os.path.join('plots',mod.name,fname.replace('.npy','.pdf')))

pbar.close()

if opts.interactive:
  plt.show()
