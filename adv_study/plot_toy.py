import os
import fnmatch

rel_info = {}

for fil in os.listdir("fitres"):
  if fnmatch.fnmatch(fil, "toyres_*.log"):
    nevs = int(fil.split('toyres_n')[1].split('.log')[0])
    rel_info[nevs] = {}
    with open(os.path.join('fitres',fil)) as f:
      for line in f.readlines():
        if line.startswith('Parameter'): continue
        if line.startswith('----'): continue
        els = line.split()
        rel_info[nevs][els[0]] = [ float(x) for x in els[1:] ]

curves = rel_info[list(rel_info.keys())[0]].keys()
npoints = sorted(rel_info.keys())

print(rel_info)
import matplotlib.pyplot as plt

cols = []
marks = ['o','x','X','^','s']

truth = {'Mean':10, 'Sigma':0.2}

for ptype in ['vals','pulls']:

  fig, ax = plt.subplots(1,1,figsize=(6,4))

  for i, c in enumerate(curves):

    x =  [ p-0.2+(i/len(curves))*0.4 for p in range(len(npoints)) ]
    ind = 0 if ptype=='vals' else 4
    offset = truth[c] if ptype=='vals' else 0
    yc = [ rel_info[x][c][ind]-offset for x in npoints ]
    ye = [ rel_info[x][c][ind+2] for x in npoints ]

    #x =  [ p+(i/len(curves))*0.5*p for p in npoints ]
    lab = r'$\mu$' if c=='Mean' else r'$\sigma$'

    ax.errorbar( x, yc, ye, label=lab, fmt='C%d%s'%(i,marks[i]), markersize=3., capsize=1.5, zorder=i+2  )

  if ptype=='vals': ax.plot( ax.get_xlim(), [0.,0.], 'k--', zorder=0 )
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
  if ptype=='vals': ax.set_ylabel('Parameter value - truth')
  else: ax.set_ylabel('Parameter pull')
  if ptype=='vals': ax.legend()
  else: ax.legend(ncol=2)
  fig.tight_layout()
  fig.savefig('figs/toy_%s.pdf'%ptype)
