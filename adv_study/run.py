import numpy as np
from scipy.stats import norm, expon, crystalball
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib import cm
import matplotlib as mpl
import pickle

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('-n','--nevents', default=10000, type=int, help='Number of events in toy')
parser.add_argument('-r','--regen', default=False, action="store_true", help='Regen the toy')
parser.add_argument('-f','--refit', default=False, action="store_true", help='Refit the toy')
parser.add_argument('-w','--rewht', default=False, action="store_true", help='Recompute the weights')
parser.add_argument('-s','--seed', default=1, type=int, help='Random seed')
parser.add_argument('-b','--batch', default=False, action="store_true", help='Run in batch mode (don\'t make plots etc.)')
parser.add_argument('-a','--all', default=False, action="store_true", help='Rerun all')
opts = parser.parse_args()

if opts.all:
  opts.regen = True
  opts.refit = True
  opts.rewht = True

print('Seed:  ', opts.seed)
print('Events:', opts.nevents)

import os
os.system('mkdir -p figs')

# ranges
mrange = (5000,6000)

# components of the PDF
compnames = ['sig','misrec1','misrec2','partreclow','partrechigh','bkg']


pdf_typs = {'sig'          : norm,
            'misrec1'      : norm,
            'misrec2'      : norm,
            'partreclow'   : crystalball,
            'partrechigh'  : crystalball,
            'bkg'          : expon
          }

pdf_vars = {'sig'          : (5350,20),
            'misrec1'      : (5350,80),
            'misrec2'      : (5330,50),
            'partreclow'   : (3 , 2 , 5000, 100),
            'partrechigh'  : (3, 2 , 5500, 100),
            'bkg'          : (5000, 400)
          }

pdf_dic = { name: pdf_typs[name](*pdf_vars[name]) for name in compnames}

# frac_yields
frac_ylds = { 'sig'          : 0.2,
              'misrec1'      : 0.05,
              'misrec2'      : 0.05,
              'partreclow'   : 0.05,
              'partrechigh'  : 0.1,
              'bkg'          : 0.4
          }

# normalise yields
norm_sum = sum(frac_ylds.values())
for name in compnames: frac_ylds[name] /= norm_sum

# set yields
abs_ylds = { name : frac_ylds[name]*opts.nevents for name in compnames }
if not opts.batch: print(abs_ylds)

# set normalisations
pdf_norms = { name : np.diff( pdf_dic[name].cdf(mrange))[0] for name in compnames }

# define a pdf function
def pdf(x, comps='all'):
    if comps=='all': comps = compnames
    return sum( [ abs_ylds[comp] * pdf_dic[comp].pdf(x) / pdf_norms[comp] for comp in comps ] )

# draw the mass model pdf(s)
if not opts.batch:
  fig, ax = plt.subplots(1,1, figsize=(6,4))
  m = np.linspace(*mrange,200)

  ax.plot(m, pdf(m), 'k-' )
  for comp in compnames: ax.plot(m, pdf(m,comps=[comp]), label=comp)
  ax.legend()
  fig.tight_layout()
  fig.savefig('figs/ex2_mass_model1.pdf')

  fig, ax = plt.subplots(1,1, figsize=(6,4))
  rcomps = list(compnames)
  for comp in compnames:
    ax.fill_between(m, pdf(m, comps=rcomps), label=comp)
    rcomps.remove(comp)
  ax.legend()
  ax.set_xlabel('mass')
  ax.set_ylabel('arbitrary units')
  fig.tight_layout()
  fig.savefig('figs/ex2_mass_model2.pdf')

# make a dalitz model
from dalitz import dalitz
mB = 5.350
mD = 1.800
mK = 0.493
mPi = 0.139

bdkp = dalitz( mB, mD, mK, mPi )

# generate a toy
import pandas as pd
np.random.seed(opts.seed)  # fix seed

os.system('mkdir -p toys')
toy_fname = 'toys/toy_n%d_s%d.pkl'%(opts.nevents,opts.seed)

if opts.regen:

  data = pd.DataFrame(columns=['mass','ctrl','m2ab','m2ac'])
  data = data.astype({'mass':float,'ctrl':int,'m2ab':float,'m2ac':float})

  for i, comp in enumerate(compnames):
      nevs = np.random.poisson(abs_ylds[comp])
      ngen = 0
      while ngen<nevs:
          mval = pdf_dic[comp].rvs()
          if mval>=mrange[0] and mval<=mrange[1]:
              ctrl = i
              if ctrl==0:
                  x = np.random.normal(10,0.2)
                  y = np.random.uniform(*bdkp.acrange)
                  while not bdkp.in_kine_limits(x,y):
                      x = np.random.normal(10,0.2)
                      y = np.random.uniform(*bdkp.acrange)
              elif ctrl==4:
                  x = np.random.uniform(*bdkp.abrange)
                  y = np.random.normal(15,0.4)
                  while not bdkp.in_kine_limits(x,y):
                      x = np.random.uniform(*bdkp.abrange)
                      y = np.random.normal(15,0.4)
              else:
                  x,y = bdkp.psgen()[0]

              data = data.append( {'mass':mval, 'ctrl': ctrl, 'm2ab':x, 'm2ac':y}, ignore_index=True )
              ngen += 1
  data.to_pickle(toy_fname)

else:
  data = pd.read_pickle(toy_fname)

if not opts.batch: print(data)

# plot the toy
if not opts.batch:
  # dal var
  fig, ax = plt.subplots(1, 1, figsize=(6,4))
  m12 = np.linspace(*bdkp.abrange,400)
  m13 = np.linspace(*bdkp.acrange,400)

  x,y = np.meshgrid(m12,m13)

  for i, comp in enumerate(compnames):
      dslice = data[data['ctrl']==i]
      ax.plot( dslice['m2ab'].to_numpy()[::int(opts.nevents/10000)], dslice['m2ac'].to_numpy()[::int(opts.nevents/10000)], '.' , markersize=0.4,  label=comp)
      ax.contour(x,y,bdkp.dp_contour(x,y,orientation=1213),[1.],colors=('b'))

  legend = ax.legend(frameon=True)
  cols = []
  for legend_handle in legend.legendHandles:
      legend_handle._legmarker.set_markersize(10)
      cols.append(legend_handle._legmarker.get_color())

  fig.tight_layout()
  fig.savefig('figs/ex2_toy_d.pdf')

  # mass var
  fig, ax = plt.subplots(1, 1, figsize=(6,4))
  ax.hist( tuple( data[data['ctrl']==i]['mass'].to_numpy() for i in reversed(range(len(compnames))) ) , bins=50, range=mrange, stacked=True, color=tuple(reversed(cols)), label=tuple(reversed(compnames)))
  ax.legend()
  fig.tight_layout()
  fig.savefig('figs/ex2_toy_m.pdf')

  # ctrl var
  fig, ax = plt.subplots(1, 1, figsize=(6,4))
  ax.hist( tuple( data[data['ctrl']==i]['ctrl'].to_numpy() for i in reversed(range(len(compnames))) ) , bins=50, range=(0,len(compnames)-1), stacked=True, color=tuple(reversed(cols)), label=tuple(reversed(compnames)))
  ax.legend()
  fig.tight_layout()
  fig.savefig('figs/ex2_toy_c.pdf')

# fit the data
# going to freeze shapes for simplicity
from iminuit import Minuit

def nll(sigy, misrec1y, misrec2y, partreclowy, partrechighy, bkgy):

    s = pdf_dic['sig']
    mr1 = pdf_dic['misrec1']
    mr2 = pdf_dic['misrec2']
    pr1 = pdf_dic['partreclow']
    pr2 = pdf_dic['partrechigh']
    b = pdf_dic['bkg']
    sn = np.diff( s.cdf(mrange))
    bn = np.diff( b.cdf(mrange))
    mr1n = np.diff( mr1.cdf(mrange))
    mr2n = np.diff( mr2.cdf(mrange))
    pr1n = np.diff( pr1.cdf(mrange))
    pr2n = np.diff( pr2.cdf(mrange))
    no = sigy + bkgy + misrec1y + misrec2y + partreclowy + partrechighy
    ne = np.sum(np.log(s.pdf(data['mass'].to_numpy()) / sn * sigy +
                       b.pdf(data['mass'].to_numpy()) / bn * bkgy +
                       mr1.pdf(data['mass'].to_numpy()) / mr1n * misrec1y +
                       mr2.pdf(data['mass'].to_numpy()) / mr2n * misrec2y +
                       pr1.pdf(data['mass'].to_numpy()) / pr1n * partreclowy +
                       pr2.pdf(data['mass'].to_numpy()) / pr2n * partrechighy
                      )
               )

    return no-ne

mikwargs = {}
for name in compnames:
    mikwargs[name+'y'] = abs_ylds[name]
    mikwargs['limit_'+name+'y'] = (0.5*abs_ylds[name],2.5*abs_ylds[name])

# fix the misrec yields as we might do in an analysis
#mikwargs['fix_misrec1y'] = True
#mikwargs['fix_misrec2y'] = True

mi = Minuit(nll, **mikwargs,
            errordef=Minuit.LIKELIHOOD,
            pedantic=False)

os.system('mkdir -p fitres')
if opts.refit:
  mi.migrad()
  mi.hesse()

  outf = open('fitres/fitres_n%d_s%d.pkl'%(opts.nevents,opts.seed),'wb')
  pickle.dump( mi.fitarg, outf )
  outf.close()

  outf = open('fitres/cov_n%d_s%d.pkl'%(opts.nevents,opts.seed),'wb')
  pickle.dump( mi.np_covariance(), outf )
  outf.close()

inf = open('fitres/fitres_n%d_s%d.pkl'%(opts.nevents,opts.seed),'rb')
fitarg = pickle.load(inf)
inf.close()
mi = Minuit(nll, **fitarg, pedantic=False)

if not opts.batch: print(mi.params)

# set the fitted yields as our yields
for comp in compnames:
    abs_ylds[comp] = mi.values[comp+'y']

# draw the fit result
if not opts.batch:
  fig, ax = plt.subplots(1,1,figsize=(6,4))
  w, xe = np.histogram( data['mass'].to_numpy(), bins=50, range=mrange )
  cx = 0.5 * (xe[1:] + xe[:-1])
  # bin width to normalise mass_pdf for plotting
  x = np.linspace(*mrange,1000)
  pdfnorm = (mrange[1]-mrange[0])/50
  rcomps = list(compnames)
  for comp in compnames:
      ax.fill_between(x, pdfnorm*pdf(x, comps=rcomps), label=comp)
      rcomps.remove(comp)
  ax.errorbar( cx, w, w**0.5, fmt='ko', label='Toy Data')
  ax.plot( x, pdfnorm*pdf(x),'b-', linewidth=2, label='Total PDF')
  ax.legend()
  ax.set_xlabel('mass')
  ax.set_ylabel('number of events')

  fig.tight_layout()
  fig.savefig('figs/ex2_fit_m.pdf')

# compute the weights
wt_fname = 'toys/toy_n%d_s%d_wts.pkl'%(opts.nevents,opts.seed)
if opts.rewht:
  import sys
  sys.path.append("/Users/matt/Scratch/stats/sweights")
  from SWeighter import SWeight

  pdfs = [ pdf_dic[comp] for comp in compnames]
  ylds = [ abs_ylds[comp] for comp in compnames]

  sw = SWeight(data['mass'].to_numpy(), pdfs=pdfs, yields=ylds, method='summation', discvarranges=(mrange,), compnames=compnames)

  # plot the weights
  if not opts.batch:
    fig, ax = plt.subplots(1,1,figsize=(6,4))
    dopts = [ c for c in cols]
    sw.makeWeightPlot(ax,dopts)
    ax.set_xlabel('mass')
    ax.set_ylabel('weight')

    fig.tight_layout()
    fig.savefig('figs/ex2_weights.pdf')

  # add the weights to the df
  for i, comp in enumerate(compnames):
      data.insert( len(data.columns), 'sw_%s'%comp, sw.getWeight(i, data['mass'].to_numpy()) )

  # save res
  data.to_pickle(wt_fname)

else:
  data = pd.read_pickle(wt_fname)

if not opts.batch: print(data)

# now look at the sweighted data
if not opts.batch:
  fig, ax = plt.subplots(1,1,figsize=(6,4))

  # going to do the histogram 7 times
  # once for the ``true" distribution and then each time
  # with a weight applied
  hdata    = [ data['ctrl'].to_numpy() for i in range(len(compnames)+1) ]
  hweights = [ np.ones(data['ctrl'].to_numpy().shape) ] + [ data['sw_%s'%comp].to_numpy() for comp in compnames]
  colors   = ['0.7'] + cols
  labels   = ['none'] + [ r'$c=%d$ (%s)'%(i+1,comp) for i, comp in enumerate(compnames) ]

  xlabpos = [ i for i in range(len(compnames)) ]
  ax.hist( hdata, bins=6, range=(-0.5,5.5), weights=hweights, color=colors, label=labels)
  ax.set_xticks(xlabpos)
  ax.set_xticklabels([x+1 for x in xlabpos])
  ax.set_xlabel(r'Control variable $c$')
  ax.set_ylabel('weighted events')
  ax.legend(title='Applied weight')

  fig.tight_layout()
  fig.savefig('figs/ex2_ctrl_wt.pdf')

  # and plot the Dalitz distributions
  from matplotlib import colors

  m12 = np.linspace(*bdkp.abrange,400)
  m13 = np.linspace(*bdkp.acrange,400)

  x,y = np.meshgrid(m12,m13)

  for i, comp in enumerate(compnames):
      fig, ax = plt.subplots(1, 1, figsize=(6,4))
      #plax = (i%2, int(i/2))
      h, xe, ye = np.histogram2d( data['m2ab'].to_numpy(), data['m2ac'].to_numpy(), weights=data['sw_%s'%comp].to_numpy(), bins=[80,80], range=(bdkp.abrange,bdkp.acrange) )
      h[h==0] = np.nan
      im = ax.imshow(h.T, interpolation='nearest', origin='low', extent=[xe[0], xe[-1], ye[0], ye[-1]], aspect='auto' )
      ax.contour(x,y,bdkp.dp_contour(x,y,orientation=1213),[1.],colors=('r'),linewidths=(3))
      ax.set_xlabel(r'$m^2_{ab}$')
      ax.set_ylabel(r'$m^2_{ac}$')
      cb = fig.colorbar(im, ax=ax)
      cb.set_label(r'weighted entries')

      fig.tight_layout()
      fig.savefig('figs/ex2_dalitz_c%d.pdf'%i)

# weighted fit to projections
from scipy.misc import derivative
from tabulate import tabulate
import sys
sys.path.append("/Users/matt/Scratch/stats/sweights")
from CovarianceCorrector import cov_correct

# plotting
table = { "Parameter" : ["Mean","MeanErr","Sigma","SigmaErr"] }
if not opts.batch: fig, ax = plt.subplots(1,1,figsize=(6,4))

for i, proj in enumerate(['sig']):#,'partrechigh']):
  pdata = None
  if proj=='sig':
    m2range = (8,12)
    m2var = 'm2ab'
    stm = 10
    sts = 0.2
  else:
    m2range = (11,19)
    m2var = 'm2ac'
    stm = 15
    sts = 0.4

  pdata   = data[ (data[m2var]>=m2range[0]) & (data[m2var]<=m2range[1]) ]

  vdata   = pdata[m2var].to_numpy()
  wts     = pdata['sw_%s'%proj].to_numpy()

  def wnll(mean, sigma):
    s = norm(mean,sigma)
    sn = np.diff(s.cdf(m2range))
    spdf = s.pdf( vdata )
    wt   = wts[ spdf>0 ]
    spdf = spdf[ spdf>0 ]
    return -np.sum( wt * np.log( spdf / sn ) )

  def mpdf(mean, sigma, x):
    s = norm(mean,sigma)
    sn = np.diff(s.cdf(m2range))
    res = s.pdf(x) / sn
    res[ res<1e-20 ] = 1e-20
    return res

  # minimisations
  mi = Minuit( wnll, mean=stm, limit_mean=m2range, sigma=sts, limit_sigma=(0.05,1.), errordef=Minuit.LIKELIHOOD, pedantic=False )
  mi.migrad()
  mi.hesse()

  fvals = mi.np_values()
  fcov   = mi.np_covariance()
  if not opts.batch: print(mi.params)

  ncov = cov_correct(mpdf, vdata, wts, fvals, fcov, verbose=False)

  table[proj] = [ fvals[0], ncov[0,0]**0.5, fvals[1], ncov[1,1]**0.5 ]

  if not opts.batch:
    nbins = 80
    w, xe = np.histogram(vdata, weights=wts, bins=nbins, range=m2range)
    cx = 0.5 * (xe[1:] + xe[:-1])
    ax.errorbar( cx, w, np.abs(w)**0.5, fmt='ko' )

    x = np.linspace(*m2range, 400)
    pdfnorm = ( m2range[1] - m2range[0] ) / nbins
    ax.plot( x, np.sum(wts)*pdfnorm*mpdf( *fvals, x), 'b-')

## print res
print(tabulate(table,headers="keys",floatfmt=".6f"))
with open('fitres/shape_n%d_s%d.txt'%(opts.nevents,opts.seed),'w') as f:
  f.write( tabulate(table, headers="keys", floatfmt=".8f") )

if not opts.batch:
  fig.tight_layout()
  fig.savefig('figs/ex2_m2dal.pdf')
