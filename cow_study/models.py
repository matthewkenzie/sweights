### Implement some classes which construct the truth models
### for the COW study
### What we need:
###
###   - a signal model which factorises in m and t (like real life)
###      - gaussian in mass, exponential in decay time
###
###   - a background model which can or cannot factorise
###      - factorising: exponential in mass and gaussian in decay time
###      - non-factori: exponential in mass (with dependence of slope on decay time)
###                     gaussian in decay time (with dependence of mean and width on mass)
###
###   - an efficiency model which doesn't factorise
###      - a custom power law like function of the form a(x+b)^a

import os
import pickle
import numpy as np
from tqdm import tqdm

from scipy.stats import truncnorm, truncexpon
from scipy.interpolate import interp1d
from scipy.integrate import quad, nquad
from scipy.optimize import minimize

from iminuit import Minuit

# some useful helper functions
def mynorm(xmin,xmax,mu,sg):
  a, b = (xmin-mu)/sg, (xmax-mu)/sg
  return truncnorm(a,b,mu,sg)

def myexp(xmin,xmax,lb):
  return truncexpon( (xmax-xmin)/lb, xmin, lb )

def mypol(xmin,xmax,po):
  return lambda x: (po+1)*(1-(x-xmin)/(xmax-xmin))**po / (xmax-xmin)

# abstract base class for models (can do some cacheing etc)
class model:
  def __init__(self, mrange, trange, name='', pars={}, cache=None):
    '''
    name: required if cache is not None as will define file to read and write cache from
    pars: required when cacheing to check parameters haven't changed
    cache: if None will not do any loading or saving of cache. if int then will cache this many points. if string will load the cache
    '''
    self.mrange = mrange
    self.trange = trange
    self.cache  = cache
    if cache is not None:
      if not isinstance(name,str): raise RuntimeError('name must be a string')
      if len(name)==0: raise RuntimeError('name must not be an empty string')
      if len(pars.keys())==0: raise RuntimeError('pars must have at least one item')
      self.name = name
      self.pars = pars
      if name.startswith('cache/'): self.name = name.replace('cache/','')
      self.cachedir = 'cache/'+self.name

  def pdf(self):
    raise NotImplementedError("Must override pdf()")

  def pdfm(self):
    raise NotImplementedError("Must override pdf()")

  def pdft(self):
    raise NotImplementedError("Must override pdf()")

  def eff(self):
    raise NotImplementedError("Must override pdf()")

  def effm(self):
    raise NotImplementedError("Must override pdf()")

  def efft(self):
    raise NotImplementedError("Must override pdf()")

  def cacheing(self):
    if self.cache is None: return

    if isinstance(self.cache,str):
      self.load_caches()
    elif isinstance(self.cache,int):
      self.make_caches(self.cache)
      self.load_caches()
    else:
      raise RuntimeError('cache option is not recognised')

  def make_caches(self, points):
    print('Caching into', self.cachedir)
    os.system(f'mkdir -p {self.cachedir}')

    # save pars
    with open(f'{self.cachedir}/pars.pkl','wb') as f:
      pickle.dump(self.pars,f)

    # compute arrays and save them
    m = np.linspace(*self.mrange,points)
    t = np.linspace(*self.trange,points)
    fm = self.pdfm(m)
    ft = self.pdft(t)
    em = self.effm(m)
    et = self.efft(t)
    np.savez(f'{self.cachedir}/arrs.npz', m=m, fm=fm, em=em, t=t, ft=ft, et=et)

  def load_caches(self):
    # read pars
    assert( os.path.exists(f'{self.cachedir}/pars.pkl') )
    with open(f'{self.cachedir}/pars.pkl','rb') as f:
      pars = pickle.load(f)
      for name, val in pars.items():
        assert( self.pars[name] == val )
    # read arrays
    assert( os.path.exists(f'{self.cachedir}/arrs.npz') )
    npzfile = np.load(f'{self.cachedir}/arrs.npz')
    m = npzfile['m']
    t = npzfile['t']
    fm = npzfile['fm']
    ft = npzfile['ft']
    em = npzfile['em']
    et = npzfile['et']
    self.pdfm = interp1d(m,fm,kind='quadratic')
    self.pdft = interp1d(t,ft,kind='quadratic')
    self.effm = interp1d(m,em,kind='quadratic')
    self.efft = interp1d(t,et,kind='quadratic')

# the efficiency model
class effmodel(model):
  def __init__(self, mrange, trange, a=0.2, b=0., dx=-0.1, cache=None, quiet=True):
    '''
      a encodes the power
      b encodes the fraction of m added to this
      dm encode the offset
    '''
    if not quiet:
      print('Initialising efficiency model')

    self.a      = a
    self.b      = b
    self.dx     = dx
    self.N      = 1
    pars = { 'a': self.a, 'b': self.b, 'dx': self.dx }

    # call parent constructor
    model.__init__(self,mrange,trange,'effmodel', pars, cache)

    # get normalisation
    self.N, self.Nerr = nquad( self.pdf, (self.mrange, self.trange) )
    if not quiet:
      print(' --> Normalisation =', self.N)

    # efficiency projections
    # and scale to the max
    self.emmax = 1
    self.etmax = 1
    self.emmax = self._effm(self.mrange[1])
    self.etmax = self._efft(self.trange[1])
    self.effm = np.vectorize(self._effm)
    self.efft = np.vectorize(self._efft)

    # pdf projections
    self.pdfm = np.vectorize(self._pdfm)
    self.pdft = np.vectorize(self._pdft)

    # enable cache
    self.cacheing()

  # the efficiency map (with max at 1)
  def eff(self,m,t):
    # get t into range [0,1]
    tpos = ( t - self.trange[0] ) / ( self.trange[1] - self.trange[0] )

    # get m into range [-1,1]
    mpos = 2 * ( m - self.mrange[0] ) / ( self.mrange[1] - self.mrange[0] ) - 1

    # scaling of factor for m -> t dependence
    ascl = self.a * (1 + self.b * mpos)

    # function
    f = ascl * ( 10*tpos - self.dx ) ** ascl

    # scale it so that the max is 1 which happens at mpos=1, tpos=1
    mx = (self.a * (1+self.b) ) * ( 10 - self.dx ) ** (self.a * (1+self.b))

    # return
    return f/mx

  # efficiency as normalised pdf
  def pdf(self, m, t):
    return self.eff(m,t) / self.N

  # 1d projections of map and pdf
  def _effm(self,m):
    f = lambda t: self.eff(m,t)
    return quad(f,*self.trange)[0] / self.emmax

  def _efft(self,t):
    f = lambda m: self.eff(m,t)
    return quad(f,*self.mrange)[0] / self.etmax

  def _pdfm(self,m):
    f = lambda t: self.pdf(m,t)
    return quad(f,*self.trange)[0]

  def _pdft(self,t):
    f = lambda m: self.pdf(m,t)
    return quad(f,*self.mrange)[0]

class sigmodel(model):
  def __init__(self, mrange, trange, mu, sg, lb, cache=None):

    print('Initialising signal model')

    self.mu     = mu
    self.sg     = sg
    self.lb     = lb
    pars = { 'mu': mu, 'sg': sg, 'lb': lb }

    # call parent constructor
    model.__init__(self,mrange,trange,'sigmodel', pars, cache)

    # can directly make the pdfs here because there is no correlation
    self.mpdf = mynorm(*self.mrange, self.mu, self.sg)
    self.tpdf = myexp(*self.trange, self.lb)

    # can directly make the pdf functions too
    self.pdf = lambda m,t: self.mpdf.pdf(m) * self.tpdf.pdf(t)
    self.pdfm = lambda m: self.mpdf.pdf(m)
    self.pdft = lambda t: self.tpdf.pdf(t)

    # need to implement eff functions but they are unity
    self._effmt = lambda m,t: 1
    self.effmt = np.vectorize(self._effmt)
    self.effm = lambda m: np.ones_like(m)
    self.efft = lambda t: np.ones_like(t)

    # call cache from parent
    self.cacheing()

  def generate(self, size=1, progress=None, save=None, seed=None):

    if progress is None:
      progress = False if size<10 else True

    if seed is not None:
      np.random.seed(seed)

    if progress: bar = tqdm(desc='Generating toy', total=size)

    vals =  np.column_stack( (self.mpdf.rvs(size=size), self.tpdf.rvs(size=size) ) )

    if progress:
      bar.update(size)
      bar.close()

    if save is not None:
      np.save(f'{save}',vals)
    else:
      return vals

class sigweffmodel(model):
  def __init__(self, mrange, trange, mu, sg, lb, ea, eb, ed, cache=None, ecache='load'):

    print('Initialising signal with efficiency model')

    self.mu     = mu
    self.sg     = sg
    self.lb     = lb
    self.ea     = ea
    self.eb     = eb
    self.ed     = ed
    pars = { 'mu': mu, 'sg': sg, 'lb': lb, 'ea': ea, 'eb': eb, 'ed': ed }

    # call parent constructor
    model.__init__(self,mrange,trange,'sigweffmodel', pars, cache)

    # can directly make the pdfs here because there is no correlation
    self.mpdf = mynorm(*self.mrange, self.mu, self.sg)
    self.tpdf = myexp(*self.trange, self.lb)

    # now make the eff map
    self.eff  = effmodel(self.mrange, self.trange, self.ea, self.eb, self.ed, ecache)
    # and implement the eff proj functions
    self.effmt = self.eff.eff
    self.effm = self.eff.effm
    self.efft = self.eff.efft

    # get normalisations
    self.N      = 1
    self.N, self.Nerr   = nquad( self.pdf, (self.mrange, self.trange) )
    print(' --> Normalisation =', self.N)

    # vectorise projection pdfs
    self.pdfm   = np.vectorize(self._pdfm)
    self.pdft   = np.vectorize(self._pdft)

    # call cache from parent
    self.cacheing()

  def pdf(self, m, t, mproj=False, tproj=False):
    eff = self.eff.eff(m,t)
    if mproj: return eff*self.mpdf.pdf(m) / self.eN
    if tproj: return eff*self.tpdf.pdf(t) / self.eN
    return eff*self.mpdf.pdf(m)*self.tpdf.pdf(t) / self.N

  def _pdfm(self,m):
    f = lambda t: self.pdf(m,t)
    return quad(f,*self.trange)[0]

  def _pdft(self,t):
    f = lambda m: self.pdf(m,t)
    return quad(f,*self.mrange)[0]

  def generate(self, size=1, progress=None, save=None, seed=None):

    # this is the very slow accept/reject method

    if progress is None:
      progress = False if size<10 else True

    if seed is not None:
      np.random.seed(seed)

    vals = np.empty((size,2))
    ngen = 0
    if progress: bar = tqdm(desc='Generating toy', total=size)
    while ngen < size:
      accept = False
      m = None
      t = None
      while not accept:
        m = self.mpdf.rvs()
        t = self.tpdf.rvs()
        h = np.random.uniform()
        if h < self.eff.eff(m,t): accept=True

      vals[ngen] = (m,t)
      if progress: bar.update()
      ngen += 1

    if progress: bar.close()
    if save is not None:
      np.save(f'{save}',vals)
    else:
      return vals

class bkgmodel(model):
  def __init__(self, mrange, trange, lb, mu, sg, cache=None):

    print('Initialising background model')

    self.lb     = lb
    self.mu     = mu
    self.sg     = sg
    pars = { 'lb': lb, 'mu': mu, 'sg': sg }

    # call parent constructor
    model.__init__(self,mrange,trange,'bkgmodel', pars, cache)

    # can directly make the pdfs here because there is no correlation
    self.mpdf = myexp(*self.mrange, self.lb)
    self.tpdf = mynorm(*self.trange, self.mu, self.sg)

    # can directly make the pdf functions too
    self.pdf = lambda m,t: self.mpdf.pdf(m) * self.tpdf.pdf(t)
    self.pdfm = lambda m: self.mpdf.pdf(m)
    self.pdft = lambda t: self.tpdf.pdf(t)

    # need to implement eff functions but they are unity
    self._effmt = lambda m,t: 1
    self.effmt = np.vectorize(self._effmt)
    self.effm = lambda m: np.ones_like(m)
    self.efft = lambda t: np.ones_like(t)

    # call cache from parent
    self.cacheing()

  def generate(self, size=1, progress=None, save=None, seed=None):

    if progress is None:
      progress = False if size<10 else True

    if seed is not None:
      np.random.seed(seed)

    if progress: bar = tqdm(desc='Generating toy', total=size)

    vals =  np.column_stack( (self.mpdf.rvs(size=size), self.tpdf.rvs(size=size) ) )

    if progress:
      bar.update(size)
      bar.close()

    if save is not None:
      np.save(f'{save}',vals)
    else:
      return vals

class bkgnfmodel(model):
  def __init__(self, mrange, trange, lb, mu, sg, slb, smu, ssg, cache=None):

    print('Initialising non-factorising background model')

    self.lb     = lb
    self.mu     = mu
    self.sg     = sg
    self.slb    = slb
    self.smu    = smu
    self.ssg    = ssg
    pars = { 'lb': lb, 'mu': mu, 'sg': sg, 'slb': slb, 'smu': smu, 'ssg': ssg }

    # call parent constructor
    model.__init__(self,mrange,trange,'bkgnfmodel', pars, cache)

    # get normalisation
    self.N      = 1
    self.N, self.Nerr   = nquad( self.pdf, (self.mrange, self.trange) )
    print(' --> Normalisation =', self.N)

    # get the majorant for toys
    f = lambda m, t: -self.pdf(m,t)
    mi = Minuit(f, m=self.mrange[0], t=self.trange[0], limit_m=self.mrange, limit_t=self.trange, pedantic=False)
    mi.migrad()
    self.maj = -mi.fval
    print(' --> Majorant of {:4.2g} found at ({:7.2f},{:4.2f})'.format(self.maj, mi.values['m'], mi.values['t']))

    # vectorise projection pdfs
    self.pdfm   = np.vectorize(self._pdfm)
    self.pdft   = np.vectorize(self._pdft)

    # need to implement eff functions but they are unity
    self._effmt = lambda m,t: 1
    self.effmt = np.vectorize(self._effmt)
    self.effm = lambda m: np.ones_like(m)
    self.efft = lambda t: np.ones_like(t)

    # call cache from parent
    self.cacheing()

  def pdf(self, m, t, mproj=False, tproj=False):

    # lambda for mass part
    dt = 2 * ( t - self.trange[0] ) / ( self.trange[1] - self.trange[0] ) - 1
    flb = self.lb + self.slb*dt
    # mu, sigma for time part
    dm = 2 * ( m - self.mrange[0] ) / ( self.mrange[1] - self.mrange[0] ) - 1
    fmu = self.mu + self.smu*dm
    fsg = self.sg + self.ssg*dm

    # pdfs
    mpdf = myexp(*self.mrange, flb)
    tpdf = mynorm(*self.trange, fmu, fsg)

    if mproj: return mpdf.pdf(m)
    if tproj: return tpdf.pdf(t)
    return mpdf.pdf(m)*tpdf.pdf(t) / self.N

  def _pdfm(self,m):
    f = lambda t: self.pdf(m,t)
    return quad(f,*self.trange)[0]

  def _pdft(self,t):
    f = lambda m: self.pdf(m,t)
    return quad(f,*self.mrange)[0]

  def generate(self, size=1, progress=None, save=None, seed=None):

    # this is the very slow accept/reject method

    if progress is None:
      progress = False if size<10 else True

    if seed is not None:
      np.random.seed(seed)

    vals = np.empty((size,2))
    ngen = 0
    if progress: bar = tqdm(desc='Generating', total=size)
    while ngen < size:
      accept = False
      m = None
      t = None
      while not accept:
        m = np.random.uniform(*self.mrange)
        t = np.random.uniform(*self.trange)
        p = self.pdf(m,t)
        if p>self.maj:
          print('Warning found p(m,t) =', p, 'at', m, ',', t, 'is larger than majorant', self.maj, '. Having to update the majorant')
          self.maj = p
        h = np.random.uniform(0,self.maj)
        if h<p: accept=True

      vals[ngen] = (m,t)
      if progress: bar.update()
      ngen += 1

    if progress: bar.close()
    if save is not None:
      np.save(f'{save}',vals)

    return vals

class bkgweffmodel(model):
  def __init__(self, mrange, trange, lb, mu, sg, ea, eb, ed, cache=None, ecache='load'):

    print('Initialising background with efficiency model')

    self.lb     = lb
    self.mu     = mu
    self.sg     = sg
    self.ea     = ea
    self.eb     = eb
    self.ed     = ed
    pars = { 'lb': lb, 'mu': mu, 'sg': sg, 'ea': ea, 'eb': eb, 'ed': ed }

    # call parent constructor
    model.__init__(self,mrange,trange,'bkgweffmodel', pars, cache)

    # can directly make the pdfs here because there is no correlation
    self.mpdf = myexp(*self.mrange, self.lb)
    self.tpdf = mynorm(*self.trange, self.mu, self.sg)

    # now make the eff map
    self.eff  = effmodel(self.mrange, self.trange, self.ea, self.eb, self.ed, ecache)
    # and implement the eff proj functions
    self.effmt = self.eff.eff
    self.effm = self.eff.effm
    self.efft = self.eff.efft

    # get normalisation
    self.N      = 1
    self.N, self.Nerr   = nquad( self.pdf, (self.mrange, self.trange) )
    self.eN     = self.eff.N
    print(' --> Normalisation =', self.N)

    # vectorise projection pdfs
    self.pdfm   = np.vectorize(self._pdfm)
    self.pdft   = np.vectorize(self._pdft)

    # call cache from parent
    self.cacheing()

  def pdf(self, m, t, mproj=False, tproj=False):
    eff = self.eff.eff(m,t)
    if mproj: return eff*self.mpdf.pdf(m) / self.eN
    if tproj: return eff*self.tpdf.pdf(t) / self.eN
    return eff*self.mpdf.pdf(m)*self.tpdf.pdf(t) / self.N

  def _pdfm(self,m):
    f = lambda t: self.pdf(m,t)
    return quad(f,*self.trange)[0]

  def _pdft(self,t):
    f = lambda m: self.pdf(m,t)
    return quad(f,*self.mrange)[0]

  def generate(self, size=1, progress=None, save=None, seed=None):

    # this is the very slow accept/reject method

    if progress is None:
      progress = False if size<10 else True

    if seed is not None:
      np.random.seed(seed)

    vals = np.empty((size,2))
    ngen = 0
    if progress: bar = tqdm(desc='Generating toy', total=size)
    while ngen < size:
      accept = False
      m = None
      t = None
      while not accept:
        m = self.mpdf.rvs()
        t = self.tpdf.rvs()
        h = np.random.uniform()
        if h < self.eff.eff(m,t): accept=True

      vals[ngen] = (m,t)
      if progress: bar.update()
      ngen += 1

    if progress: bar.close()
    if save is not None:
      np.save(f'{save}',vals)
    else:
      return vals

class bkgnfweffmodel(model):
  def __init__(self,mrange, trange, lb, mu, sg, slb, smu, ssg, ea, eb, ed, cache=None, ecache='load'):

    print('Initialising non-factorising background with efficiency model')

    self.lb     = lb
    self.mu     = mu
    self.sg     = sg
    self.slb    = slb
    self.smu    = smu
    self.ssg    = ssg
    self.ea     = ea
    self.eb     = eb
    self.ed     = ed
    pars = { 'lb': lb, 'mu': mu, 'sg': sg, 'slb': slb, 'smu': smu, 'ssg': ssg, 'ea': ea, 'eb': eb, 'ed': ed }

    # call parent constructor
    model.__init__(self,mrange,trange,'bkgnfweffmodel', pars, cache)

    # now make the eff map
    self.eff  = effmodel(self.mrange, self.trange, self.ea, self.eb, self.ed, ecache)
    # and implement the eff proj functions
    self.effmt = self.eff.eff
    self.effm = self.eff.effm
    self.efft = self.eff.efft

    # get normalisation
    self.N      = 1
    self.N, self.Nerr   = nquad( self.pdf, (self.mrange, self.trange) )
    self.eN     = self.eff.N
    print(' --> Normalisation =', self.N)

    # get the majorant for toys
    f = lambda m, t: -self.pdf(m,t)
    mi = Minuit(f, m=self.mrange[0], t=self.trange[0], limit_m=self.mrange, limit_t=self.trange, pedantic=False)
    mi.migrad()
    self.maj = -mi.fval
    print(' --> Majorant of {:4.2g} found at ({:7.2f},{:4.2f})'.format(self.maj, mi.values['m'], mi.values['t']))

    # vectorise projection pdfs
    self.pdfm   = np.vectorize(self._pdfm)
    self.pdft   = np.vectorize(self._pdft)

    # call cache from parent
    self.cacheing()

  def pdf(self, m, t, mproj=False, tproj=False):

    eff = self.eff.eff(m,t)

    # lambda for mass part
    dt = 2 * ( t - self.trange[0] ) / ( self.trange[1] - self.trange[0] ) - 1
    flb = self.lb + self.slb*dt
    # mu, sigma for time part
    dm = 2 * ( m - self.mrange[0] ) / ( self.mrange[1] - self.mrange[0] ) - 1
    fmu = self.mu + self.smu*dm
    fsg = self.sg + self.ssg*dm

    # pdfs
    mpdf = myexp(*self.mrange, flb)
    tpdf = mynorm(*self.trange, fmu, fsg)

    if mproj: return eff*mpdf.pdf(m) / self.eN
    if tproj: return eff*tpdf.pdf(t) / self.eN
    return eff*mpdf.pdf(m)*tpdf.pdf(t) / self.N

  def _pdfm(self,m):
    f = lambda t: self.pdf(m,t)
    return quad(f,*self.trange)[0]

  def _pdft(self,t):
    f = lambda m: self.pdf(m,t)
    return quad(f,*self.mrange)[0]

  def generate(self, size=1, progress=None, save=None, seed=None):

    # this is the very slow accept/reject method

    if progress is None:
      progress = False if size<10 else True

    if seed is not None:
      np.random.seed(seed)

    vals = np.empty((size,2))
    ngen = 0
    if progress: bar = tqdm(desc='Generating', total=size)
    while ngen < size:
      accept = False
      m = None
      t = None
      while not accept:
        m = np.random.uniform(*self.mrange)
        t = np.random.uniform(*self.trange)
        p = self.pdf(m,t)
        if p>self.maj:
          print('Warning found p(m,t) =', p, 'at', m, ',', t, 'is larger than majorant', self.maj, '. Having to update the majorant')
          self.maj = p
        h = np.random.uniform(0,self.maj)
        if h<p: accept=True

      vals[ngen] = (m,t)
      if progress: bar.update()
      ngen += 1

    if progress: bar.close()
    if save is not None:
      np.save(f'{save}',vals)

    return vals

