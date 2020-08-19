from model import multibkgmodel
from argparse import ArgumentParser
import sys
sys.path.append('..')
from MySWeightClass import SWeight

parser = ArgumentParser()
parser.add_argument('-n','--ntoys'  , default=1, type=int,                help='Number of toys' )
parser.add_argument('-e','--nevents', default=10000, type=int,            help='Number of events per toy')
parser.add_argument('-d','--dir'    , default='toys',                     help='Directory to save toy output in')
parser.add_argument('-r','--readgen', default=False, action="store_true", help='Just read the toys from files already written')
parser.add_argument('-R','--readfit', default=False, action="store_true", help='Read the fit result from files already written')
parser.add_argument('-g','--genonly', default=False, action="store_true", help='Just generate the toys nothing else')
parser.add_argument('-c','--swcheck', default=False, action="store_true", help='Run sWeight checks')
parser.add_argument('-s','--startat', default=1, type=int,                help='Start number for toys')
parser.add_argument('-b','--batch'  , default=False, action="store_true", help='Run in batch mode')
opts = parser.parse_args()

import matplotlib as mpl
if opts.batch: mpl.use('pdf')
import matplotlib.pyplot as plt

import os
os.system('mkdir -p %s'%opts.dir)

for it in range(opts.startat,opts.ntoys+opts.startat):
  os.system('mkdir -p %s/toy%d'%(opts.dir,it))

  mbm = multibkgmodel( nevents=opts.nevents )

  # Generate toy
  if not opts.readgen:
    print('Generating toy {:>4d} / {:<4d}'.format(it,opts.ntoys+opts.startat-1), end='')
    toy = mbm.generate( save='%s/toy%d/gen.pkl'%(opts.dir,it), seed=it )
    print(' - {0} events'.format(len(toy)))

  if opts.genonly: continue

  toy = mbm.read_toy( '%s/toy%d/gen.pkl'%(opts.dir,it) )

  # Fit toy for yields
  if not opts.readfit:
    print('Fitting toy {:>4d} / {:<4d}'.format(it,opts.ntoys+opts.startat-1))
    mbm.fit(toy, save='%s/toy%d/fit.pkl'%(opts.dir,it))

  mbm.load_fit_res( '%s/toy%d/fit.pkl'%(opts.dir,it) )

  # Run the sweights
  print('Computing weights {:>4d} / {:<4d}'.format(it,opts.ntoys+opts.startat-1))
  pdfs = [ mbm.pdf_dic[comp] for comp in mbm.compnames ]
  ylds = [ mbm.abs_ylds[comp] for comp in mbm.compnames ]
  sw = SWeight(toy['mass'].to_numpy(), pdfs=pdfs, yields=ylds, method='summation', discvarranges=(mbm.mrange,), compnames=mbm.compnames, checks=opts.swcheck)

  # Add the weights to the dataframe
  for i, comp in enumerate(mbm.compnames):
    toy.insert( len(toy.columns), 'w'+comp, sw.getWeight(i, toy['mass'].to_numpy()) )
  toy.to_pickle('%s/toy%d/weights.pkl'%(opts.dir,it))

  # Save the weights in the frame
  fig, ax = plt.subplots( 1, 2, figsize=(12,4) )

  # Draw the fit result
  mbm.draw(axis=ax[0], npoints=1000, dset=toy, nbins=50)
  ax[0].set_xlabel('Mass [MeV/$c^{2}$]')
  ax[0].set_ylabel('Number of Events')
  ax[0].set_xlim( 5000, 6000 )
  ax[0].set_ylim( 0, ax[0].get_ylim()[1] )
  ax[0].set_title('Toy %d Fit Result'%it)

  # Draw the weight distributions
  dopts = plt.rcParams['axes.prop_cycle'].by_key()['color']
  sw.makeWeightPlot( ax[1], dopts )
  ax[1].set_xlabel('Mass [MeV/$c^{2}$]')
  ax[1].set_ylabel('Weight')
  ax[1].set_title('Toy %d sWeights'%it)
  if not opts.batch: fig.tight_layout()
  fig.savefig('%s/toy%d/toy.pdf'%(opts.dir,it))

  fig.clear()
  plt.close(fig)
