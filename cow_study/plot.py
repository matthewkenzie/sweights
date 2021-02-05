import numpy as np
import pickle
import matplotlib.pyplot as plt

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('-v','--vals',default=False, action="store_true", help='Use values instead of pull')
opts = parser.parse_args()

pull = not opts.vals
offset = 0.05

def read_pull_file(fname):
  with open(fname,'rb') as f:
    res = pickle.load(f)
  eqev = res['Equiv']['vals']
  if pull:
    lmbdc = res['lmbdc']['pull']
    lmbdr = res['lmbdr']['pull']
  else:
    lmbdc = res['lmbdc']['vals']
    lmbdr = res['lmbdr']['vals']
  return lmbdc, lmbdr, eqev

def pulls_as_line( files ):
  x = np.array(list(range(1, len(files)+1)))
  yc = np.array( [ read_pull_file(f)[0] for f in files ] )
  yr = np.array( [ read_pull_file(f)[1] for f in files ] )
  ee = np.array( [ read_pull_file(f)[2] for f in files ] )
  return x, yc, yr, ee

def add_pulls_to_axis( files, ax, bkgs, **kwargs ):
  x, yc, yr, ee = pulls_as_line( files )
  if pull:
    ax.fill_between( [0.5, len(files)+1.5], -1, 1, alpha=0.5, color='0.75' )
    ax.plot( [0.5, len(files)+1.5], [0,0], 'k--' )
  else:
    ax.plot( [0.5, len(files)+1.5], [4,4], 'k--' )

  if len(bkgs)<=1:
    ax.errorbar( x-offset, yc[:,0,0], yc[:,1,0], **kwargs, marker='^', ls='', capsize=2, label='Corr. Error' )
    ax.errorbar( x+offset, yr[:,0,0], yr[:,1,0], **kwargs, marker='o', ls='', capsize=2, label='Raw Error' )
    ylim = ax.get_ylim()
    yval = 0.95*ylim[1]
    for i in range(len(x)): ax.text( x[i], yval, r'${:.1f} \pm {:.1f}$'.format(ee[i,0,0],ee[i,1,0]), ha='center' )
  else:
    xstart = -2*(len(bkgs)-1)*offset
    for ib, bkg in enumerate(bkgs):
      nfiles = []
      for fil in files:
        if '_cow' in fil:
          els = fil.split('_')
          bval = None
          for el in els:
            if el.startswith('B'):
              bval = int(el.replace('B',''))
          nfiles.append( fil.replace('B%d'%bval,'B%d'%bkg) )
        else:
          nfiles.append( fil )
      x, yc, yr, ee = pulls_as_line( nfiles )
      if ib>0:
        x = x[1:]
        yc = yc[1:]
        yr = yr[1:]

      dx = xstart+ib*4*offset
      if bkg==-1: labext = ' - $g_{b}(m) = g_{b}(m)$'
      else: labext = ' - $g_{b}(m)$ = pol%d'%bkg
      ax.errorbar( x+dx-offset, yc[:,0,0], yc[:,1,0], **kwargs, marker='^', ls='', capsize=2, label='Corr. Error'+labext )
      #ax.errorbar( x+dx-offset, yc[:,0,0], yc[:,0,1], **kwargs, marker='^', ls='', capsize=2, label=None )
      ax.errorbar( x+dx+offset, yr[:,0,0], yr[:,1,0], **kwargs, marker='o', ls='', capsize=2, label='Raw Error'+labext )
      ylim = ax.get_ylim()
      yval = (0.95-ib*0.1)*ylim[1]
      for i in range(len(x)): ax.text( x[i]+dx, yval, r'${:.1f} \pm {:.1f}$'.format(ee[i,0,0],ee[i,1,0]), ha='center' )

def style_axis( ax ):
  ax.set_xticks( range(1,5) )
  ax.set_xticklabels( ['SW','COW $I(m)=1$', 'COW $I(m)=f(m)$', 'COW $I(m)=q(m)$'] )
  ax.set_xlim( 0.5, 4.5 )
  ylim = ax.get_ylim()
  if pull:
    y = max(abs(ylim[0]),abs(ylim[1]))
    ax.set_ylim(-y,y)
    ax.set_ylabel('pull $(\lambda)$')
  else:
    y = max(abs(ylim[1]-4),abs(ylim[0]-4))
    ax.set_ylim(4-y,4+y)
    ax.set_ylabel(r'$\langle \lambda \rangle$')

fig, ax = plt.subplots(2,2,figsize=(18,10))

# read factorising no efficiency
ax[0,0].set_title('No efficiency, factorising signal and background')
add_pulls_to_axis( [ 'res/sig_n1000_bkg_n1000/pull_res_sw.pkl',
                     'res/sig_n1000_bkg_n1000/pull_res_cow_S0_B-1_I1.pkl',
                     'res/sig_n1000_bkg_n1000/pull_res_cow_S0_B-1_I2.pkl',
                     'res/sig_n1000_bkg_n1000/pull_res_cow_S0_B-1_I3.pkl' ],
                   ax[0,0],
                   bkgs=[-1])
ax[0,0].legend()
style_axis(ax[0,0])


# read factorising with efficiency
ax[1,0].set_title('With efficiency, factorising signal and background')
add_pulls_to_axis( [ 'res/sigweff_n1000_bkgweff_n1000/pull_res_sw_eff.pkl',
                     'res/sigweff_n1000_bkgweff_n1000/pull_res_cow_eff_S0_B-1_I1.pkl',
                     'res/sigweff_n1000_bkgweff_n1000/pull_res_cow_eff_S0_B-1_I2.pkl',
                     'res/sigweff_n1000_bkgweff_n1000/pull_res_cow_eff_S0_B-1_I3.pkl' ],
                   ax[1,0],
                   bkgs=[-1,2])
ax[1,0].legend()
style_axis(ax[1,0])

# read non-factorising no efficiency
ax[0,1].set_title('No efficiency, factorising signal and non-factorising background')
add_pulls_to_axis( [ 'res/sig_n1000_bkgnf_n1000/pull_res_sw.pkl',
                     'res/sig_n1000_bkgnf_n1000/pull_res_cow_S0_B-1_I1.pkl',
                     'res/sig_n1000_bkgnf_n1000/pull_res_cow_S0_B-1_I2.pkl',
                     'res/sig_n1000_bkgnf_n1000/pull_res_cow_S0_B-1_I3.pkl' ],
                   ax[0,1],
                   bkgs=[-1,2,3,4])
ax[0,1].legend()
style_axis(ax[0,1])

# read non-factorising with efficiency
ax[1,1].set_title('With efficiency, factorising signal and non-factorising background')
add_pulls_to_axis( [ 'res/sigweff_n1000_bkgnfweff_n1000/pull_res_sw_eff.pkl',
                     'res/sigweff_n1000_bkgnfweff_n1000/pull_res_cow_eff_S0_B-1_I1.pkl',
                     'res/sigweff_n1000_bkgnfweff_n1000/pull_res_cow_eff_S0_B-1_I2.pkl',
                     'res/sigweff_n1000_bkgnfweff_n1000/pull_res_cow_eff_S0_B-1_I3.pkl' ],
                   ax[1,1],
                   bkgs=[-1,2,3,4])
ax[1,1].legend()
style_axis(ax[1,1])

fig.tight_layout()
if pull:
  fig.savefig('plots/cow_results_pull.pdf')
else:
  fig.savefig('plots/cow_results_vals.pdf')
plt.show()
