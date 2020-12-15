import os
import sys
sys.path.append("/Users/matt/Scratch/stats/sweights")
import numpy as np
import pandas as pd

from scipy.stats import norm, expon, uniform
from scipy.integrate import quad

import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Slider

from toy import toy
from cow import cow
from iminuit import Minuit
from CovarianceCorrector import cov_correct

class cowlab:
  def __init__(self):
    print('starting cowlab')
    self.mrange = (5000,5600)
    self.m = np.linspace(*self.mrange, 200)
    self.trange = (0,10)
    self.t = np.linspace(*self.trange, 200)

    # set some defaults
    self.poln = 0
    self.bins = 50

    # set the plottable objects to None
    self.Im = None
    self.gs = None
    self.gb = None
    self.data = None
    self.obs = None
    self.cow = None
    self.eff = None

    # set up some functions
    self.sig = norm(5280,30)
    self.sn  = np.diff( self.sig.cdf(self.mrange) )
    self.bkg = expon(self.mrange[0],400)
    self.bn  = np.diff( self.bkg.cdf(self.mrange) )
    self.unf = uniform(*self.mrange)
    self.un  = np.diff( self.unf.cdf(self.mrange) )
    self.g0  = lambda m: self.sig.pdf(m) / self.sn
    self.g1  = lambda m: self.bkg.pdf(m) / self.bn
    self.rho = lambda m: 0.5*self.g0(m) + 0.5*self.g1(m)
    self.fold = lambda m: (m-self.mrange[0])/(self.mrange[1]-self.mrange[0])
    self.pn  = lambda n, m: (n+1)*self.fold(m)**n / np.diff(self.mrange)
    self.p0  = lambda m: 1*self.fold(m)**0 / np.diff(self.mrange)
    self.p1  = lambda m: 2*self.fold(m)**1 / np.diff(self.mrange)
    self.p2  = lambda m: 3*self.fold(m)**2 / np.diff(self.mrange)
    self.p3  = lambda m: 4*self.fold(m)**3 / np.diff(self.mrange)
    self.p4  = lambda m: 5*self.fold(m)**4 / np.diff(self.mrange)
    self.p5  = lambda m: 6*self.fold(m)**5 / np.diff(self.mrange)
    self.p6  = lambda m: 7*self.fold(m)**6 / np.diff(self.mrange)
    self.pols = [ self.p0, self.p1, self.p2, self.p3, self.p4, self.p5, self.p6 ]

    # set the defaults
    self.setIm(1)
    self.setGs(2)
    self.setGb(-1)
    self.setEff('flat')

  def setIm(self, typ):
    if   typ==1: self.Im = lambda m: self.unf.pdf(m) / self.un
    elif typ==2: self.Im = lambda m: self.rho(m)
    else: raise RuntimeError('Unrecognised option for I(m)')

  def setGs(self, typ):
    if   typ==1: self.gs = lambda m: self.unf.pdf(m) / self.un
    elif typ==2: self.gs = lambda m: self.g0(m)
    else: raise RuntimeError('Unrecognised option for gs')

  def setGb(self, typ):
    if   typ==-1: self.gb = [ self.g1 ]
    elif typ>=0: self.gb = [ self.pols[n] for n in range(typ+1) ]
    else: raise RuntimeError('Unrecognised option for gb')

  def setEff(self, label):
    if label not in ['flat','fact','nonfact']:
      raise RuntimeError('Not a valid option for efficiency', label)
    self.eff = label

  def readData(self):
    toyfname = 'toys/toy_e{0}_bfact_10000.pkl'.format(self.eff)
    print('Reading data toy from', toyfname)
    if not os.path.exists(toyfname):
      raise RuntimeError('No file here', toyfname)
    self.data = pd.read_pickle( toyfname )

  # nll function for weighted data
  def wnll(self, lambd):
    b = expon(self.trange[0], lambd)
    bn = np.diff( b.cdf(self.trange) )
    return -np.sum( self.dsw * ( b.logpdf( self.data['time'] ) - np.log(bn) ) )

  def timepdf(self, lambd, t):
    b = expon(self.trange[0], lambd)
    bn = np.diff( b.cdf(self.trange) )
    return b.pdf(t) / bn

  def wfit(self):
    self.dsw = self.cow.wk(0,self.data['mass'])
    mi = Minuit( self.wnll, lambd=2, errordef=0.5, pedantic=False )
    mi.migrad()
    mi.hesse()
    cov = cov_correct(self.timepdf, self.data['time'], self.dsw, mi.np_values(), mi.np_covariance(), verbose=False)
    self.fval = mi.values['lambd']
    self.ferr = cov[0,0]**0.5

  def plot_pdfs(self, ax):
    # do stuff
    ax.clear()
    if self.gs is not None:
      ax.plot(self.m, np.diff(self.mrange)*self.gs(self.m), 'b-', label='signal',zorder=10)
    if self.gb is not None:
      for i, gb in enumerate(self.gb):
        label = 'background'
        if len(self.gb)>1: label += ' p{0}'.format(i)
        ax.plot( self.m, np.diff(self.mrange)*gb(self.m), 'r-'.format(i), label=label, zorder=i )
    if self.gs is not None or self.gb is not None:
      ax.legend()
    ax.set_title('Component PDFs ($g_{k}$s)')
    ax.set_xlabel('mass')
    ax.set_ylabel('probability')

  def plot_wts(self, ax):
    # do stuff
    ax.clear()
    if self.cow is not None:
      self.sw = self.cow.wk(0,self.m)
      ax.plot( self.m, self.sw, 'b-', label='signal' )
      self.bw = np.sum( [ self.cow.wk(i+1,self.m) for i in range(len(self.gb)) ], axis=0 )
      ax.plot( self.m, self.bw, 'r-', label='background' )
      ax.plot( self.m, self.sw + self.bw, 'k-', label='sum')
      ax.legend()
    ax.set_title('Weight distributions')
    ax.set_xlabel('mass')
    ax.set_ylabel('weight')

  def plot_tdata(self, ax):
    # do stuff
    ax.clear()
    if self.cow is not None:
      self.wfit()
      w, xe = np.histogram( self.data['time'], bins=self.bins, range=self.trange, weights=self.dsw )
      cx = 0.5 * (xe[1:] + xe[:-1] )
      ax.errorbar( cx, w, w**0.5, fmt='bx', label='sCOW weigted data' )
      pnorm = np.sum(self.dsw)*np.diff(self.trange)/self.bins
      ax.plot( self.t, pnorm*self.timepdf(2, self.t), 'b-', label='True $h_0(t)$ distribution' )
      ax.plot( self.t, pnorm*self.timepdf(self.fval,self.t), 'r--', label='Fitted $h_0(t)$ distribution' )
      ax.set_yscale('log')
      ylim = ax.get_ylim()
      ax.set_ylim( 1, ylim[1] )
      # info
      truen = len(self.data[self.data['ctrl']==0])
      sum_w = np.sum(self.dsw)
      err_w = np.sum(self.dsw**2)**0.5
      ax.text(0.32,0.9,'$\lambda = {:.2f} \pm {:.2f}$'.format( self.fval, self.ferr ), transform=ax.transAxes )
      ax.text(0.6,0.7,'$\sum w = {:.2f} \pm {:.2f}$'.format( sum_w,err_w ), transform=ax.transAxes )
      ax.text(0.6,0.62,'True $N_{{s}} = {{{0}}}$'.format( truen ), transform=ax.transAxes )
      ax.legend()

    ax.set_title('COW weighted time distribution')
    ax.set_xlabel('time')
    ax.set_ylabel('weighted events')

  def plot_mdata(self, ax):
    # do stuff
    ax.clear()
    pnorm = np.diff(self.mrange)
    if self.data is not None:
      w, xe = np.histogram( self.data['mass'], bins=self.bins, range=self.mrange )
      cx = 0.5 * (xe[1:] + xe[:-1] )
      ax.errorbar( cx, w, w**0.5, fmt='bx', label=r'Observed data')
      pnorm = np.sum(w)*np.diff(self.mrange)/self.bins
    if self.Im is not None:
      ax.plot( self.m, pnorm*self.Im(self.m), 'r-', label=r'$I(m)$ model' )
    if self.data is not None or self.Im is not None:
      ax.legend()
    ax.set_title('Observed mass distribution')
    ax.set_xlabel('mass')
    ax.set_ylabel('events')

  def plot_model(self, fig, ax):
    ax.clear()
    if hasattr(self,'mcolb'): self.mcolb.remove()
    self.toy = toy(eff=self.eff)
    x, y = np.meshgrid(self.m,self.t)
    cb = ax.contourf( x, y, 1e3*self.toy.fmt(x,y) )
    self.mcolb = fig.colorbar( cb, ax=ax)
    self.mcolb.set_label(r'Probability $(\times 10^{-3})$')
    ax.set_title('Truth model: $f(m,t)$')
    ax.set_xlabel('mass')
    ax.set_ylabel('time')

  def plot_eff(self, fig, ax):
    ax.clear()
    if hasattr(self,'ecolb'): self.ecolb.remove()
    self.toy = toy(eff=self.eff)
    x, y = np.meshgrid(self.m,self.t)
    cb = ax.contourf( x, y, self.toy.effmt(x,y), levels=np.linspace(0,1,11) )
    self.ecolb = fig.colorbar( cb, ax=ax)
    self.ecolb.set_label('Efficiency')
    ax.set_title('Efficiency map')
    ax.set_xlabel('mass')
    ax.set_ylabel('time')

  def plot_mproj(self, ax):
    ax.clear()
    self.toy = toy(eff=self.eff)
    w, xe = np.histogram( self.data['mass'], bins=self.bins, range=self.mrange )
    norm = np.sum(w) * np.diff(self.mrange)/self.bins
    if self.eff!='flat': norm /= quad(lambda m: self.toy.rhomt(m, None), *self.mrange)[0]
    cx = 0.5 * (xe[1:]+xe[:-1])
    ax.errorbar( cx, w, w**0.5, fmt='ko', ms=3, capsize=2, label='Toy Data' )
    by = self.toy.rhomt(self.m, None, bonly=True)
    ty = self.toy.rhomt(self.m, None )
    ax.plot( self.m, norm*by, 'r--', label='Background')
    ax.plot( self.m, norm*ty, 'b-', label='Signal + Background')
    ax.legend()
    ax.set_title('Truth model with efficiency projection in mass')
    ax.set_xlabel('mass')

  def plot_tproj(self, ax):
    ax.clear()
    self.toy = toy(eff=self.eff)
    w, xe = np.histogram( self.data['time'], bins=self.bins, range=self.trange )
    norm = np.sum(w) * np.diff(self.trange)/self.bins
    cx = 0.5 * (xe[1:]+xe[:-1])
    ax.errorbar( cx, w, w**0.5, fmt='ko', ms=3, capsize=2, label='Toy Data' )
    ax.set_title('Truth model with efficiency projection in time')
    ax.set_ylabel('time')

  def update(self):
    # read in the toy (or make it)
    self.readData()
    # run the cow
    self.cow = cow(mrange=self.mrange, gs=self.gs, gb=self.gb, Im=self.Im, obs=self.obs)
    print( self.cow.Wkl() )
    print( self.cow.Akl() )
    self.plot_pdfs(self.cax[0,0])
    self.plot_wts(self.cax[1,0])
    self.plot_tdata(self.cax[0,1])
    self.plot_mdata(self.cax[1,1])
    self.plot_model(self.tfig,self.tax[0,0])
    self.plot_eff(self.tfig,self.tax[1,0])
    self.plot_mproj(self.tax[0,1])
    self.plot_tproj(self.tax[1,1])
    self.tfig.tight_layout()
    self.tfig.canvas.draw_idle()
    self.cfig.tight_layout()
    self.cfig.canvas.draw_idle()

  def run(self):

    # create figs
    self.tfig, self.tax = plt.subplots(2,2,figsize=(12,8))
    self.cfig, self.cax = plt.subplots(2,2,figsize=(12,8))

    # widget axis
    wfig = plt.figure(figsize=(2,4))

    # Im widget
    butIm = RadioButtons(plt.axes([0.,0.8,1,0.2]), ('$I(m)=1$',r'$I(m)=\rho(m)$', '$I(m)=q(m)$'), active=0 )
    def imfunc(label):
      if   label == '$I(m)=1$': self.setIm(1)
      elif label == r'$I(m)=\rho(m)$': self.setIm(2)
      elif label == '$I(m)=q(m)$': self.setIm(3)
      self.update()
    butIm.on_clicked(imfunc)

    # Gs widget
    butGs = RadioButtons(plt.axes([0,0.6,1,0.2]), ('$g_s(m)=1$',r'$g_s(m)=g_0$', '$g_s(m)=p(m)$'), active=1 )
    def gsfunc(label):
      if   label == '$g_s(m)=1$': self.setGs(1)
      elif label == r'$g_s(m)=g_0$': self.setGs(2)
      elif label == '$g_s(m)=p(m)$': self.setGs(3)
      self.update()
    butGs.on_clicked(gsfunc)

    # Gb widget
    butGb = RadioButtons(plt.axes([0,0.4,1,0.2]), ('exp','pols'), active=0 )
    def gbfunc(label):
      if   label == 'exp': self.setGb(-1)
      elif label == 'pols': self.setGb(self.poln)
      self.update()
    butGb.on_clicked(gbfunc)

    # Gb slider
    sldGb = Slider(plt.axes([0.2,0.3,0.6,0.05]), 'nBkg', 0,6,valstep=1,valinit=0)
    def sgbfunc(val):
      self.setGb(sldGb.val)
      self.poln = sldGb.val
      butGb.set_active(1)
      self.update()
    sldGb.on_changed(sgbfunc)

    # Gb text box

    # Efficiency
    butEff = RadioButtons(plt.axes([0,0,1,0.2]), ('flat','fact','nonfact'), active=0)
    def efffunc(label):
      self.setEff(label)
      self.update()
    butEff.on_clicked(efffunc)

    # update things
    self.update()
    plt.show()


    # do the plots
    #self.plot_pdfs(self.ax1)
    #self.plot_wts(self.ax2)
    #self.plot_tdata(self.ax3)
    #self.plot_mdata(self.ax4)


cl = cowlab()
a = cl.run()

