# A class for a multibkg model

from scipy.stats import norm, crystalball, expon
import matplotlib.pyplot as plt
import numpy as np
from dalitz import dalitz
import pandas as pd
from iminuit import Minuit
import pickle

class multibkgmodel():
  def __init__(self, nevents=10000, mrange=(5000,6000) ):

    self.nevents = nevents
    self.mrange  = mrange

    self.compnames = ['sig','misrec1','misrec2','partreclow','partrechigh','bkg']

    self.pdf_typs = {'sig'          : norm,
                     'misrec1'      : norm,
                     'misrec2'      : norm,
                     'partreclow'   : crystalball,
                     'partrechigh'  : crystalball,
                     'bkg'          : expon
                    }

    self.pdf_vars = {'sig'          : (5350,20),
                     'misrec1'      : (5350,80),
                     'misrec2'      : (5330,50),
                     'partreclow'   : (3, 2 , 5000, 100),
                     'partrechigh'  : (3, 2 , 5500, 100),
                     'bkg'          : (5000, 400)
                    }

    self.frac_ylds = { 'sig'          : 0.2,
                       'misrec1'      : 0.05,
                       'misrec2'      : 0.05,
                       'partreclow'   : 0.05,
                       'partrechigh'  : 0.1,
                       'bkg'          : 0.4
                     }

    # normalise yields
    self.norm_sum = sum(self.frac_ylds.values())
    for name in self.compnames: self.frac_ylds[name] /= self.norm_sum

    # set yields
    self.abs_ylds = { name : self.frac_ylds[name]*self.nevents for name in self.compnames }

    ### build the pdfs ###
    # creates self.pdf_dic
    # creates self.pdf_norms
    self.build_tot_pdf()

		# make a dalitz object for control class
		# let's assume B0 -> D0 K+ pi-
    mB = 5.350
    mD = 1.800
    mK = 0.493
    mPi = 0.139
    self.dalitz = dalitz( mB, mD, mK, mPi )


  def build_tot_pdf(self):

    # build component pdfs
    self.pdf_dic = { name: self.pdf_typs[name](*self.pdf_vars[name]) for name in self.compnames}

    # store their normalisation in the range
    self.pdf_norms = { name : np.diff( self.pdf_dic[name].cdf(self.mrange))[0] for name in self.compnames }

  def pdf(self, x, comps='all'):
    if comps=='all': comps = self.compnames
    return sum( [ self.abs_ylds[name] * self.pdf_dic[name].pdf(x) / self.pdf_norms[name] for name in comps ] )

  def draw(self, axis=None, dset=None, npoints=200, nbins=100, stacked=True):

    ax = axis or plt.gca()

    x = np.linspace( *self.mrange, npoints )

    rcomps = list(self.compnames)

    pdfnorm = 1.
    if dset is not None:
      pdfnorm = (self.mrange[1]-self.mrange[0])/nbins

    if stacked:
      for i, comp in enumerate(self.compnames):
        ax.fill_between( x, pdfnorm*self.pdf(x, comps=rcomps), label=comp, zorder=i )
        rcomps.remove(comp)
    else:
      for i, comp in enumerate(self.compnames):
        ax.plot( x, pdfnorm*self.pdf(x, comps=[comp]), label=comp, zorder=i )

    ax.plot( x, pdfnorm*self.pdf(x), 'b-', label='Total PDF', zorder=i+1)

    if dset is not None:
      if isinstance(dset, pd.DataFrame):
        dvals = dset['mass'].to_numpy()
      elif isinstance(dset, np.array):
        dvals = dset
      else:
        raise TypeError("Cannot recognise the type of the dataset you're trying to fit")
      w, xe = np.histogram( dvals, bins=nbins, range=self.mrange )
      cx = 0.5 * (xe[1:] + xe[:-1])
      ex = 0.5 * (xe[1:] - xe[:-1])
      ax.errorbar( cx, w, w**0.5, ex, fmt='ko', label='Data', elinewidth=1., markersize=3., capsize=1.5, zorder=i+2)

    ax.legend()

  def generate(self, seed=None, poisson=True, save=None):

    np.random.seed(seed)

    self.toy = pd.DataFrame(columns=['mass','ctrl','m2ab','m2ac'])

    for i, comp in enumerate(self.compnames):
      nevs = self.abs_ylds[comp]
      if poisson: nevs = np.random.poisson(self.abs_ylds[comp])
      ngen = 0
      while ngen<nevs:
        mval = self.pdf_dic[comp].rvs()
        if mval>=self.mrange[0] and mval<=self.mrange[1]:
          ctrl = i
          if ctrl==0:
            x = np.random.normal(10,0.2)
            y = np.random.uniform(*self.dalitz.acrange)
            while not self.dalitz.in_kine_limits(x,y):
              x = np.random.normal(10,0.2)
              y = np.random.uniform(*self.dalitz.acrange)
          elif ctrl==4:
            x = np.random.uniform(*self.dalitz.abrange)
            y = np.random.normal(15,0.4)
            while not self.dalitz.in_kine_limits(x,y):
              x = np.random.uniform(*self.dalitz.abrange)
              y = np.random.normal(15,0.4)
          else:
            x,y = self.dalitz.psgen()[0]

          self.toy = self.toy.append( {'mass':mval, 'ctrl': ctrl, 'm2ab':x, 'm2ac':y}, ignore_index=True )
          ngen += 1

    self.toy = self.toy.astype({'mass':float,'ctrl':int,'m2ab':float,'m2ac':float})
    if save is not None:
      self.toy.to_pickle(save)

    return self.toy

  def read_toy(self, fname):
    self.toy = pd.read_pickle(fname)
    return self.toy

  def nll(self, pars):

    cache_ylds = dict( self.abs_ylds )
    cache_pars = dict( self.pdf_vars )

    ## set the yields and parameters
    ipar = 0
    nexp = 0
    for comp in self.compnames:
      ## yield
      self.abs_ylds[comp] = pars[ipar]
      nexp += pars[ipar]
      ipar += 1
      ## shape pars
      ##for var in self.pdf_vars[comp]:

    self.build_tot_pdf()
    nll = nexp - np.sum( np.log( self.pdf(self.fitset) ) )

    self.abs_ylds = cache_ylds
    self.pdf_vars = cache_pars

    return nll

  def fit(self, dset=None, save=None):

    if dset is None: dset = self.toy

    if isinstance(dset, pd.DataFrame):
      dvals = dset['mass'].to_numpy()
    elif isinstance(dset, np.array):
      dvals = dset
    else:
      raise TypeError("Cannot recognise the type of the dataset you're trying to fit")

    self.fitset = dset

    start_vals = []
    mi_kwargs = {}
    mi_kwargs['name'] = []
    mi_kwargs['limit'] = []
    mi_kwargs['error'] = []
    mi_kwargs['fix']  = [False, True, True, True, True, False]

    for comp in self.compnames:
      start_vals.append( self.abs_ylds[comp] )
      mi_kwargs['name'].append( comp+'_y' )
      mi_kwargs['limit'].append( (0.*self.abs_ylds[comp], 2.5*self.abs_ylds[comp] ) )
      mi_kwargs['error'].append( self.abs_ylds[comp]**0.5 )

    mi = Minuit.from_array_func( self.nll, start_vals, **mi_kwargs,
                                  errordef=Minuit.LIKELIHOOD,
                                  pedantic=False )

    mi.migrad()
    mi.hesse()
    print( mi.get_param_states())

    # set the values back to the minimum
    for comp in self.compnames:
      self.abs_ylds[comp] = mi.values[comp+'_y']

    self.build_tot_pdf()

    if save is not None:
      outf = open( save, 'wb' )
      pickle.dump( mi.fitarg, outf )
      outf.close()

  def load_fit_res(self, fname):
    inf = open( fname, 'rb' )
    fitarg = pickle.load( inf )
    inf.close()

    # get fitargs in right format for "from_array_func" call
    start_vals = []
    mi_kwargs = {}
    mi_kwargs['name'] = []
    mi_kwargs['limit'] = []
    mi_kwargs['error'] = []
    mi_kwargs['fix'] = []

    for comp in self.compnames:
      start_vals.append( fitarg[comp+'_y'] )
      mi_kwargs['name'].append( comp+'_y' )
      mi_kwargs['limit'].append( fitarg['limit_'+comp+'_y'] )
      mi_kwargs['error'].append( fitarg['error_'+comp+'_y'] )
      mi_kwargs['fix'].append( fitarg['fix_'+comp+'_y'] )

    mi = Minuit.from_array_func( self.nll, start_vals, **mi_kwargs,
                                  errordef=Minuit.LIKELIHOOD,
                                  pedantic=False )

    print( mi.get_param_states())

    # set the values to the minimum
    for comp in self.compnames:
      self.abs_ylds[comp] = mi.values[comp+'_y']

    self.build_tot_pdf()

