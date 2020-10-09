import os
import fnmatch
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('-w','--width', default=False, action="store_true", help='Use the width of the fitted values for the error bar. The default uses the average of the variance square rooted')
opts = parser.parse_args()

rel_info = {}

for fil in os.listdir("fitres"):
  if fnmatch.fnmatch(fil, "toyres_m*.log"):
    nevs = int(fil.split('_n')[1].split('.log')[0])
    mbins = int(fil.split('_m')[1].split('_n')[0])
    if nevs not in rel_info.keys(): rel_info[nevs] = {}
    with open(os.path.join('fitres',fil)) as f:
      for line in f.readlines():
        if line.startswith('Parameter'): continue
        if line.startswith('----'): continue
        els = line.split()
        rel_info[nevs][mbins] = [ float(x) for x in els[1:] ]

npoints = sorted(rel_info.keys())
mbins = sorted(rel_info[npoints[-1]].keys())

print('This is what I\'m going to plot')
for point in npoints:
  for mbin in mbins:
    if mbin not in rel_info[point].keys():
      #print('No',mbin,'for',point)
      continue
    else:
      vals = rel_info[point][mbin]
      print('%-6d'%point, '%-5d'%mbin, vals)

def points_from_bins(mbin):
  ret = []
  for p in npoints:
    if mbin in rel_info[p].keys():
      ret.append(p)
  return ret


import matplotlib.pyplot as plt
import matplotlib.patches as patches

cols = []
marks = ['o','D','v','^','s']

for ptype in ['vals','pulls']:

  fig, ax = plt.subplots(1,1,figsize=(6,4))

  for i, c in enumerate(mbins):

    points = points_from_bins(c)
    x =  [ npoints.index(x)-0.2+(i/len(mbins))*0.4 for x in points ]
    ind = 0 if ptype=='vals' else 4
    ym = [ rel_info[x][c][ind] for x in points ]
    yme = [ rel_info[x][c][ind+1] for x in points ]
    yw = [ rel_info[x][c][ind+2] for x in points ]
    ywe = [ rel_info[x][c][ind+3] for x in points ]

    if ptype=='vals' and not opts.width:
      yw  = [ rel_info[x][c][8]**0.5 for x in points ]
      ywe = [ 0.5*rel_info[x][c][8]**-0.5 * rel_info[x][c][9] for x in points ]

    #x =  [ p+(i/len(mbins))*0.5*p for p in points ]
    lab = '%d bins'%c

    ax.errorbar( x, ym, yme, fmt='C%d%s'%(i,marks[i]), linewidth=1, elinewidth=1, markersize=1., capsize=1.5, capthick=1 , zorder=i+2  )
    ax.errorbar( x, ym, yw, label=lab, fmt='C%d%s'%(i,marks[i]), linewidth=0.5, elinewidth=0.5, markersize=1., capsize=1.5, capthick=0.5, zorder=i+2+len(mbins)  )

    # draw the errors on the errors
    if ptype!='vals':
      wd = 0.05
      for p in range(len(points)):
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
  #ax.set_xticks(points)
  ax.set_xticks([p for p in range(len(npoints))])
  ax.set_xticklabels([str(x) for x in npoints])
  ax.set_xlabel('Sample size')
  if ptype=='vals': ax.set_ylabel('Slope parameter value')
  else: ax.set_ylabel('Slope parameter pull')
  if ptype=='vals': ax.legend()
  else: ax.legend(ncol=2)
  fig.tight_layout()
  fig.savefig('figs/ex3_toy_%s.pdf'%ptype)
