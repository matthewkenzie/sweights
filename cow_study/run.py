from scipy.stats import norm, expon
from bernsteins import bpoly1, bpoly2, bpoly3, bpoly4
import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit

if __name__ == '__main__':
  mrange = (5000,5600)
  trange = (0,10)
  g0 = norm(5280,30)
  g1 = expon(5000,400)
  h0 = expon(0,2)
  h1 = norm(0,0.5)
  z0 = 0.5
  N = 10000

  import pandas as pd
  np.random.seed(210187)

  # generate toy
  generate = False
  fit = False
  if generate:
    data = pd.DataFrame(columns=['mass','time','ctrl'])

    n_sig = np.random.poisson(N*z0)
    n_bkg = np.random.poisson(N*(1-z0))

    ns = 0
    while ns < n_sig:
      m = g0.rvs()
      t = h0.rvs()
      if m > mrange[1] or m < mrange[0]: continue
      if t > trange[1] or t < trange[0]: continue
      data = data.append( {'mass': m, 'time': t, 'ctrl': 0}, ignore_index=True )
      ns += 1

    nb = 0
    while nb < n_bkg:
      m = g1.rvs()
      t = h1.rvs()
      if m > mrange[1] or m < mrange[0]: continue
      if t > trange[1] or t < trange[0]: continue
      data = data.append( {'mass': m, 'time': t, 'ctrl': 1}, ignore_index=True )
      nb += 1

    data = data.astype( {'mass': float, 'time': float, 'ctrl': int} )
    data.to_pickle('toy.pkl')

  else:
    data = pd.read_pickle('toy.pkl')

  # draw toy
  fig, ax = plt.subplots(1,2, figsize=(12,4))
  sig = data[data['ctrl']==0]
  bkg = data[data['ctrl']==1]

  ax[0].hist( [bkg['mass'],sig['mass']], bins=50, stacked=True )
  ax[1].hist( [bkg['time'],sig['time']], bins=50, stacked=True )
  ax[1].set_yscale('log')
  fig.tight_layout()

  def nll_exp(N, z, mu, sg, lb):
    s = norm(mu,sg)
    b = expon(mrange[0], lb)
    sn = np.diff( s.cdf(mrange) )
    bn = np.diff( b.cdf(mrange) )
    ns = N*z
    nb = N*(1-z)
    return N - np.sum( np.log ( s.pdf( data['mass'].to_numpy() ) / sn * ns + b.pdf( data['mass'].to_numpy() ) / bn * nb ) )

  def nll_poly(pars):
    N  = pars[0]
    z  = pars[1]
    mu = pars[2]
    sg = pars[3]
    s = norm(mu,sg)
    sn = np.diff( s.cdf(mrange) )
    n = len(pars[4:])
    if n==1: b  = bpoly1(1,pars[4],5000,600)
    if n==2: b  = bpoly2(1,pars[4],pars[5],5000,600)
    if n==3: b  = bpoly3(1,pars[4],pars[5],pars[6],5000,600)
    if n==4: b  = bpoly4(1,pars[4],pars[5],pars[6],pars[7],5000,600)

    bn = np.diff( b.cdf(mrange) )
    ns = N*z
    nb = N*(1-z)
    return N - np.sum( np.log ( s.pdf( data['mass'].to_numpy() ) / sn * ns + b.pdf ( data['mass'].to_numpy() ) / bn * nb ) )

  mi = Minuit( nll_exp, N=N, z=z0, mu=5280, sg=30, lb=400, errordef=0.5, pedantic=False )

  if fit:
    mi.migrad()
    mi.hesse()
    print(mi.params)

  mips =  [ Minuit.from_array_func( nll_poly, (N,z0,5280,30,0.5)            , name=('N','z','mu','sg','p1')               , limit=((0.5*N,1.5*N),(0,1),(5200,5300),(0,50),(0,1)), errordef=0.5, pedantic=False ),
            Minuit.from_array_func( nll_poly, (N,z0,5280,30,0.5,0.5)        , name=('N','z','mu','sg','p1','p2')          , limit=((0.5*N,1.5*N),(0,1),(5200,5300),(0,50),(0,1),(0,1)), errordef=0.5, pedantic=False ),
            Minuit.from_array_func( nll_poly, (N,z0,5280,30,0.5,0.5,0.5)    , name=('N','z','mu','sg','p1','p2','p3')     , limit=((0.5*N,1.5*N),(0,1),(5200,5300),(0,50),(0,1),(0,1),(0,1)), errordef=0.5, pedantic=False ),
            Minuit.from_array_func( nll_poly, (N,z0,5280,30,0.5,0.5,0.5,0.5), name=('N','z','mu','sg','p1','p2','p3','p4'), limit=((0.5*N,1.5*N),(0,1),(5200,5300),(0,50),(0,1),(0,1),(0,1),(0,1)), errordef=0.5, pedantic=False ) ]

  if fit:
    for mip in mips:
      mip.migrad()
      mip.hesse()
      print(mip.params)

# draw pdf
  def pdf(m, bonly=False, sonly=False, poly=None):

    values = mi.values if poly is None else mips[poly-1].values

    mu = values['mu']
    sg = values['sg']
    spdf = norm( mu, sg )
    sn   = np.diff(spdf.cdf(mrange))

    if poly is None:
      bpdf = expon( 5000, mi.values['lb'] )
      bn   = np.diff(bpdf.cdf(mrange))
    else:
      assert( type(poly)==int )
      args = [ values['p%d'%(p+1)] for p in range(poly) ]
      if poly == 1: bpdf = bpoly1(1,*args,5000,600)
      if poly == 2: bpdf = bpoly2(1,*args,5000,600)
      if poly == 3: bpdf = bpoly3(1,*args,5000,600)
      if poly == 4: bpdf = bpoly4(1,*args,5000,600)
      bn   = np.diff(bpdf.cdf(mrange))

    ns   = values['N']*values['z']
    nb   = values['N']*(1-values['z'])
    sr = spdf.pdf(m)
    br = bpdf.pdf(m)
    if sonly: return ns * sr / sn
    if bonly: return nb * br / bn
    return ns * sr / sn + nb * br / bn

  fig = plt.figure()
  ax = fig.gca()
  bins = 50
  pn = (mrange[1]-mrange[0])/bins
  x = np.linspace(*mrange,400)

  w, xe = np.histogram( data['mass'].to_numpy(), bins=50, range=mrange )
  cx = 0.5 * (xe[1:] + xe[:-1] )

  ax.errorbar( cx, w, w**0.5, fmt='ko')
  #ax.plot( x, pn*pdf(x,bonly=True), 'r--')
  ax.plot( x, pn*pdf(x))
  #ax.plot( x, pn*pdf(x,bonly=True,poly=1), 'g--')
  ax.plot( x, pn*pdf(x,poly=1))
  #ax.plot( x, pn*pdf(x,bonly=True,poly=2), 'g--')
  ax.plot( x, pn*pdf(x,poly=2))
  ax.plot( x, pn*pdf(x,poly=3))
  ax.plot( x, pn*pdf(x,poly=4))

  fig.tight_layout()
  #plt.show()

  from cow import cow
  for ctype in [1,2,3]:
    obs = None
    bins_obs = 200
    if ctype==3: obs = np.histogram( data['mass'], bins=bins_obs, range=mrange )
    mycow = cow(Im=ctype, obs=obs)
    print( mycow.Wkl() )
    print( mycow.Akl() )

    # fit the weighted data
    wts = mycow.wk(0,data['mass'])
    def wnll(lambd):
      b = expon(trange[0], lambd)
      bn = np.diff( b.cdf(trange) )
      return -np.sum( wts * ( b.logpdf( data['time'] ) - np.log(bn) ) )
    def timepdf(lambd,x):
      b = expon(trange[0], lambd)
      bn = np.diff( b.cdf(trange) )
      return b.pdf(x) / bn

    mit = Minuit( wnll, lambd=2, errordef=0.5, pedantic=False )
    mit.migrad()
    mit.hesse()
    import sys
    sys.path.append("/Users/matt/Scratch/stats/sweights")
    from CovarianceCorrector import cov_correct
    cov = cov_correct(timepdf, data['time'], wts, mit.np_values(), mit.np_covariance(), verbose=False)
    fval = mit.values['lambd']
    ferr = cov[0,0]**0.5

    fig, ax = plt.subplots(2,2,figsize=(12,8))

    # plot the component pdfs used for the weights
    m = np.linspace(*mrange,200)
    ax[0,0].plot( m, mycow.fmt(m,sonly=True), 'b-', label='signal')
    ax[0,0].plot( m, mycow.fmt(m,bonly=True), 'r-', label='background')
    ax[0,0].legend()
    ax[0,0].set_xlabel('mass')
    ax[0,0].set_ylabel('probability')

    # plot the weights
    sw = mycow.wk(0,m)
    bw = mycow.wk(1,m)
    ax[1,0].plot( m, sw, label='signal' )
    ax[1,0].plot( m, bw, label='background' )
    ax[1,0].plot( m, sw+bw, label='sum' )
    ax[1,0].set_xlabel('mass')
    ax[1,0].set_ylabel('weight')

    # plot the weighted data
    w, xe = np.histogram( data['time'], bins=bins, range=trange, weights=wts )
    cx = 0.5 * (xe[1:] + xe[:-1] )
    ax[0,1].errorbar( cx, w, w**0.5, fmt='bx', label='sCOW weigted data' )
    t = np.linspace(*trange,200)
    pnorm = np.sum(wts)*np.diff(trange)/bins
    ax[0,1].plot( t, pnorm*h0.pdf(t), 'b-', label='True $h_0(t)$ distribution' )
    ax[0,1].plot( t, pnorm*timepdf(fval,t), 'r--', label='Fitted $h_0(t)$ distribution' )
    ax[0,1].set_yscale('log')
    ylim = ax[0,1].get_ylim()
    ax[0,1].set_ylim( 1, ylim[1] )
    ax[0,1].set_xlabel('time')
    ax[0,1].set_ylabel('weighted events')

    truen = len(data[data['ctrl']==0])
    sum_w = np.sum(wts)
    err_w = np.sum(wts**2)**0.5
    #print(truen, truen**0.5, sum_w, err_w)
    ax[0,1].text(0.6,0.7,'$\sum w = {:.2f} \pm {:.2f}$'.format( sum_w,err_w ), transform=ax[0,1].transAxes )
    ax[0,1].text(0.6,0.6,'$\lambda = {:.2f} \pm {:.2f}$'.format( fval, ferr ), transform=ax[0,1].transAxes )
    ax[0,1].legend()

    # plot the observed data with Im
    w, xe = np.histogram( data['mass'], bins=bins, range=mrange )
    cx = 0.5 * (xe[1:] + xe[:-1] )
    ax[1,1].errorbar( cx, w, w**0.5, fmt='bx', label=r'Observed data')
    pnorm = np.sum(w)*np.diff(mrange)/bins
    if ctype==3: pnorm = np.sum(w) * (bins_obs/bins)
    ax[1,1].plot( m, pnorm*mycow.Im(m), 'r-', label=r'$I(m)$ model' )
    ax[1,1].set_xlabel('mass')
    ax[1,1].set_ylabel('events')
    ax[1,1].legend()
    fig.tight_layout()
    fig.savefig('plots/cow%d.pdf'%ctype)

  plt.show()

