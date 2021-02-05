import os

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--sm', choices=['sig','sigweff'], help='Signal toy')
parser.add_argument('--bm', choices=['bkg','bkgweff','bkgnf','bkgnfeff'], help='Background toy')
parser.add_argument('--sN', type=int, help='Signal events')
parser.add_argument('--bN', type=int, help='Background events')
parser.add_argument('-n','--ntoys', type=int, help='Number of toys')
parser.add_argument('-m','--method', choices=['sw','cow'], help='Method')
parser.add_argument('-e','--eff'   , default=False, action="store_true", help='Include efficiency effects' )
parser.add_argument('--gs', default=0 , type=int, help='Choice of gs')
parser.add_argument('--gb', default=-1, type=int, help='Choice of gb')
parser.add_argument('--Im', default=1 , type=int, help='Choice of Im')
opts = parser.parse_args()

fwd_opts = ''
if opts.eff: fwd_opts += ' -e'
if opts.method=='cow':
  fwd_opts += ' -c'
  fwd_opts += ' -S %d'%opts.gs
  if opts.gb>=0: fwd_opts += ' -p %d'%opts.gb
  else: fwd_opts += ' -B %d'%(-opts.gb-1)
  fwd_opts += ' -I %d'%opts.Im

outloc = f'{opts.sm}_n{opts.sN}_{opts.bm}_n{opts.bN}/'
outloc += f'res_{opts.method}'
if opts.eff: outloc += f'_eff'
if opts.method=='cow':
  outloc += f'_S{opts.gs}_B{opts.gb}_I{opts.Im}'

import subprocess
from multiprocessing import Pool

def run_cmd(cmd):
  print('Running:', cmd)
  subprocess.run(cmd, shell=True, check=True)

def parallel(cmds):
  pool = Pool(processes=4)
  [ pool.map( run_cmd, cmds ) ]

if __name__ == '__main__':

  cmds = []

  for it in range(1,opts.ntoys+1):
    sf = os.path.join('toys', opts.sm+'model', 'toy_n%d_t%d.npy'%(opts.sN,it))
    bf = os.path.join('toys', opts.bm+'model', 'toy_n%d_t%d.npy'%(opts.bN,it))

    if not os.path.exists(sf):
      print('Warning no sig file at', sf, 'skipping toy', it)
      continue
    if not os.path.exists(bf):
      print('Warning no bkg file at', bf, 'skipping toy', it)
      continue

    cmd = f'python run.py -s {sf} -b {bf} -o {outloc}_t{it} {fwd_opts} -q'
    cmds.append(cmd)

  parallel(cmds)




