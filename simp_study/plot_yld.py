import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

ref = 'YldYieldOnlyFitResult'
comps = ['YldVariantA','YldVariantB','YldVariantC','YldVariantD']

def getStats(nevs):
  fname = 'fitres/toyanalysis_n%d.pkl'%nevs
  assert( os.path.exists(fname) )
  data = pd.read_pickle(fname)

  rows = []

  for i, name in enumerate(['sy','sye','by','bye']):
    #fig = plt.gcf()
    #ax = plt.gca()
    for yld in comps:
      ref_vals = np.stack(data[ref].values)
      yld_vals = np.stack(data[yld].values)

      rdiff = 100*(yld_vals[:,i] - ref_vals[:,i])/ref_vals[:,i]
      mean = np.mean(rdiff)
      sdev = np.std(rdiff)
      merr = sdev / len(rdiff)**0.5
      serr = sdev / (2*len(rdiff) - 2)**0.5
      rows.append( [name, yld, mean, merr, sdev, serr ] )

      #diff = yld_vals[:,i] - ref_vals[:,i]
      #ax.hist( diff, label=lab, density=True, bins=100, range=(-1,1), zorder=orders[i] )

    #if i==0: ax.legend()
    #fig.tight_layout()
    #fig.savefig('figs/toy_%s.pdf'%name)
    #fig.clear()

  return rows

from tabulate import tabulate

marks = ['o','D','v','^','s']
smpsize = [1000,2500,5000,10000,25000]
stats   = [ getStats(ev) for ev in smpsize ]

fig, axg = plt.subplots(2,2,figsize=(12,8),sharex='col',gridspec_kw={'wspace':0,'hspace':0})

for i, name in enumerate(['sy','sye','by','bye']):

  #fig, ax = plt.subplots(1,1,figsize=(6,4))
  plax = (int(i/2), i%2)
  ax = axg[plax]

  for j, comp in enumerate(comps):

    ind = i*4 + j
    x   = [p-0.15+(j/len(comps))*0.4 for p in range(len(smpsize))]
    ym  = [ stat[ind][2] for stat in stats ]
    yme = [ stat[ind][3] for stat in stats ]
    yw  = [ stat[ind][4] for stat in stats ]
    ywe = [ stat[ind][5] for stat in stats ]

    lab = comp.replace('Yld','')
    lab = lab.replace('VariantC','VariantCi')
    lab = lab.replace('VariantD','VariantCii')
    lab = lab.replace('Variant','Variant ')

    ax.errorbar( x, ym, yme, fmt='C%d%s'%(j,marks[j]), linewidth=1, elinewidth=1, markersize=2., capsize=1.5, capthick=1 , zorder=j+2  )
    ax.errorbar( x, ym, yw, label=lab, fmt='C%d%s'%(j,marks[j]), linewidth=0.5, elinewidth=0.5, markersize=2., capsize=1.5, capthick=0.5, zorder=j+2+len(comps)  )

    # draw the errors on the errors
    wd = 0.05
    for p in range(len(smpsize)):
      ht = ywe[p]
      xp = x[p]-0.5*wd
      y1 = ym[p]+yw[p]-0.5*ht
      y2 = ym[p]-yw[p]-0.5*ht
      rect1 = patches.Rectangle( (xp,y1), wd, ht, linewidth=0, edgecolor='C%d'%j, facecolor='C%d'%j, alpha=0.5)
      rect2 = patches.Rectangle( (xp,y2), wd, ht, linewidth=0, edgecolor='C%d'%j, facecolor='C%d'%j, alpha=0.5)
      ax.add_patch(rect1)
      ax.add_patch(rect2)

  if i==0: ax.legend(fontsize=12)
  #print( [ p for p in range(len(smpsize)) ] )
  ax.set_xlim(-0.5,4.5)
  ax.plot( ax.get_xlim(), [0.,0.], 'k--', linewidth=1, zorder=0 )
  ax.minorticks_off()
  ax.autoscale(enable=True, axis='x', tight=True)
  #ax.set_xticks(npoints)
  ax.set_xticks([p for p in range(len(smpsize))])
  ax.set_xticklabels([str(x) for x in smpsize])
  ax.set_ylim( -max(*np.abs(ax.get_ylim())), max(*np.abs(ax.get_ylim())) )
  ax.tick_params(labelsize=12)
  #ax.set_ylim(-0.05,0.05)
  if i==2 or i==3:
    ax.set_xlabel('Sample size',fontsize=12)
  if i==1 or i==3:
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
  #ax.set_ylabel('Percentage difference between \n fit result yield and sum of weights')
  if name=='sy':    ax.set_ylabel(r'$\left( \sum w_{s} - N_{s}^{\mathrm{fit}}\right) / N_{s}^{\mathrm{fit}}$ [%]',fontsize=12)
  elif name=='by':  ax.set_ylabel(r'$\left( \sum w_{b} - N_{b}^{\mathrm{fit}}\right) / N_{b}^{\mathrm{fit}}$ [%]',fontsize=12)
  elif name=='sye': ax.set_ylabel(r'$\left( \sqrt{\sum w_{s}^{2}} - \sigma_{N_{s}^{\mathrm{fit}}}\right) / \sigma_{N_{s}^{\mathrm{fit}}}$ [%]',fontsize=12)
  elif name=='bye': ax.set_ylabel(r'$\left( \sqrt{\sum w_{b}^{2}} - \sigma_{N_{b}^{\mathrm{fit}}}\right) / \sigma_{N_{b}^{\mathrm{fit}}}$ [%]',fontsize=12)
  #fig.tight_layout()
  #fig.savefig('figs/yld_%s.pdf'%name)

fig.align_ylabels()
fig.tight_layout()
fig.savefig('figs/yld.pdf')

for ev in smpsize:
  rows = getStats(ev)
  print('Sample size', ev)
  print( tabulate(rows, headers=['Par','Var','Mean','MeanErr','Width','WidthErr']) )
