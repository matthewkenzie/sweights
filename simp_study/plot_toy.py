import os
import fnmatch

rel_info = {}

for fil in os.listdir("fitres"):
  if fnmatch.fnmatch(fil, "toyres_*.log"):
    nevs = int(fil.split('toyres_n')[1].split('.log')[0])
    rel_info[nevs] = {}
    with open(os.path.join('fitres',fil)) as f:
      for line in f.readlines():
        if line.startswith('Slp') and '1DTruth' not in line:
          els = line.split()
          rel_info[nevs][els[0].replace('Slp','')] = [ float(x) for x in els[1:] ]

curves = rel_info[list(rel_info.keys())[0]].keys()
npoints = sorted(rel_info.keys())

print(rel_info)
import matplotlib.pyplot as plt
import matplotlib.patches as patches

cols = []
marks = ['o','D','v','^','s']

for ptype in ['vals','pulls']:

  fig, ax = plt.subplots(1,1,figsize=(6,4))

  for i, c in enumerate(curves):

    x =  [ p-0.2+(i/len(curves))*0.4 for p in range(len(npoints)) ]
    ind = 0 if ptype=='vals' else 4
    ym = [ rel_info[x][c][ind] for x in npoints ]
    yme = [ rel_info[x][c][ind+1] for x in npoints ]
    yw = [ rel_info[x][c][ind+2] for x in npoints ]
    ywe = [ rel_info[x][c][ind+3] for x in npoints ]

    #x =  [ p+(i/len(curves))*0.5*p for p in npoints ]
    lab = c.replace('VariantC','VariantCi')
    lab = lab.replace('VariantD','VariantCii')
    lab = lab.replace('2DFitResult','2D Fit Result')
    lab = lab.replace('Variant','Variant ')

    ax.errorbar( x, ym, yme, fmt='C%d%s'%(i,marks[i]), linewidth=1, elinewidth=1, markersize=1., capsize=1.5, capthick=1 , zorder=i+2  )
    ax.errorbar( x, ym, yw, label=lab, fmt='C%d%s'%(i,marks[i]), linewidth=0.5, elinewidth=0.5, markersize=1., capsize=1.5, capthick=0.5, zorder=i+2+len(curves)  )

    # draw the errors on the errors
    wd = 0.05
    for p in range(len(npoints)):
      ht = ywe[p]
      xp = x[p]-0.5*wd
      y1 = ym[p]+yw[p]-0.5*ht
      y2 = ym[p]-yw[p]-0.5*ht
      rect1 = patches.Rectangle( (xp,y1), wd, ht, linewidth=0, edgecolor='C%d'%i, facecolor='C%d'%i, alpha=0.5)
      rect2 = patches.Rectangle( (xp,y2), wd, ht, linewidth=0, edgecolor='C%d'%i, facecolor='C%d'%i, alpha=0.5)
      ax.add_patch(rect1)
      ax.add_patch(rect2)


  if ptype=='vals': ax.plot( ax.get_xlim(), [2.,2.], 'k--', linewidth=0.5, zorder=0, label='Truth' )
  else:
    ax.plot( ax.get_xlim(), [0.,0.], 'k--', zorder=1 )
    ax.fill_between( ax.get_xlim(), [-1.,-1.], [1.,1.], color='0.75', alpha=0.5, zorder=0 )
    ax.set_ylim(-1.2,2)
  #ax.set_yscale('log')
  ax.minorticks_off()
  ax.autoscale(enable=True, axis='x', tight=True)
  #ax.set_xticks(npoints)
  ax.set_xticks([p for p in range(len(npoints))])
  ax.set_xticklabels([str(x) for x in npoints])
  ax.set_xlabel('Sample size')
  if ptype=='vals': ax.set_ylabel('Slope parameter value')
  else: ax.set_ylabel('Slope parameter pull')
  if ptype=='vals': ax.legend()
  else: ax.legend(ncol=2)
  fig.tight_layout()
  fig.savefig('figs/toy_%s.pdf'%ptype)
