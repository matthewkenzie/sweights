import numpy as np

from scipy.stats import norm, expon, uniform

import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons

class cowlab:
  def __init__(self):
    print('starting cowlab')
    self.mrange = (5000,5600)
    self.m = np.linspace(*self.mrange, 200)

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

  def setIm(self, typ):
    if   typ==1: self.Im = lambda m: self.unf.pdf(m) / self.un
    elif typ==2: self.Im = lambda m: self.rho(m)
    else: raise RuntimeError('Unrecognised option for I(m)')

  def setGk(self, typ):
    if   typ==1: self.gk = lambda m: self.unf.pdf(m) / self.un
    elif typ==2: self.gk = lambda m: self.g0(m)
    else: raise RuntimeError('Unrecognised option for gk')

  def plot_pdfs(self, ax):
    # do stuff
    ax.clear()
    if hasattr(self,'gk'):
      ax.plot(self.m, self.gk(self.m), 'b-', label='signal')
      ax.legend()
    ax.set_title('Component PDFs ($g_{k}$s)')
    ax.set_xlabel('mass')
    ax.set_ylabel('probability')

  def plot_wts(self, ax):
    # do stuff
    ax.set_title('Weight distributions')
    ax.set_xlabel('mass')
    ax.set_ylabel('weight')

  def plot_tdata(self, ax):
    # do stuff
    ax.set_title('COW weighted time distribution')
    ax.set_xlabel('time')
    ax.set_ylabel('weighted events')

  def plot_mdata(self, ax):
    # do stuff
    ax.clear()
    if hasattr(self,'Im'):
      ax.plot( self.m, self.Im(self.m) )
    ax.set_title('Observed mass distribution')
    ax.set_xlabel('mass')
    ax.set_ylabel('events')

  def update(self):
    self.plot_pdfs(self.ax[0,0])
    self.plot_wts(self.ax[0,1])
    self.plot_tdata(self.ax[1,0])
    self.plot_mdata(self.ax[1,1])
    self.fig.tight_layout()
    self.fig.canvas.draw_idle()

  def run(self):
    #self.fig = plt.figure(figsize=(15,8))
    #gs = self.fig.add_gridspec(2,5)
    #self.ax1 = self.fig.add_subplot(gs[0,0:2])
    #self.ax2 = self.fig.add_subplot(gs[1,0:2])
    #self.ax3 = self.fig.add_subplot(gs[0,2:4])
    #self.ax4 = self.fig.add_subplot(gs[1,2:4])
    self.fig, self.ax = plt.subplots(2,2,figsize=(12,8))

    # widget axis
    wfig = plt.figure(figsize=(2,4))

    # Im widget
    butIm = RadioButtons(plt.axes([0.,0.5,1,0.5]), ('$I(m)=1$',r'$I(m)=\rho(m)$', '$I(m)=q(m)$'), (False,False,False) )
    def imfunc(label):
      if   label == '$I(m)=1$': self.setIm(1)
      elif label == r'$I(m)=\rho(m)$': self.setIm(2)
      elif label == '$I(m)=q(m)$': self.setIm(3)
      self.update()
    butIm.on_clicked(imfunc)

    # Gk widget
    butGk = RadioButtons(plt.axes([0,0,1,0.5]), ('$g_s(m)=1$',r'$g_s(m)=g_0$', '$g_s(m)=p(m)$'), (False,False,False) )
    def gkfunc(label):
      if   label == '$g_s(m)=1$': self.setGk(1)
      elif label == r'$g_s(m)=g_0$': self.setGk(2)
      elif label == '$g_s(m)=p(m)$': self.setGk(3)
      self.update()
    butGk.on_clicked(gkfunc)

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

