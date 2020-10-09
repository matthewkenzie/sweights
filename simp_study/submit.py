from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('-e','--nevents', type=int, help='Number of events per toy')
parser.add_argument('-z','--zval'   , type=float, help='z value for signal to background ratio')
parser.add_argument('-n','--ntoys'  , type=int, help='Number of toys to run')
parser.add_argument('-r','--regen'  , default=False, action="store_true", help='Regenerate the toys')
parser.add_argument('-f','--refit'  , default=False, action="store_true", help='Refit the toys')
parser.add_argument('-w','--rewht'  , default=False, action="store_true", help='Recompute the weights')
parser.add_argument('-a','--all'    , default=False, action="store_true", help='Rerun all')
parser.add_argument('-s','--starttoy', type=int, default=1, help='Starting number of toys')
opts = parser.parse_args()

if opts.all:
  opts.regen = True
  opts.refit = True
  opts.rewht = True

import subprocess
from multiprocessing import Pool

from tqdm import tqdm
#pbar = tqdm( total=opts.ntoys )

def run_command(cmd):
  print('Running:', cmd)
  #pbar.update(1)
  return subprocess.run(cmd, shell=True, capture_output=False)

def submit(cmds):
  pool = Pool()
  [ pool.map( run_command, cmds ) ]
  pool.close()

cmds = []
for i in range(opts.starttoy, opts.ntoys+opts.starttoy):
  cmd = 'python run.py -z {0:4.2f} -n {1} -s {2} -b'.format(opts.zval,opts.nevents,i)
  if opts.regen: cmd += ' -r'
  if opts.refit: cmd += ' -f'
  if opts.rewht: cmd += ' -w'
  cmd += ' > logs/out_z{0:4.2f}_n{1}_s{2}.log'.format(opts.zval,opts.nevents,i)
  cmds.append(cmd)

submit(cmds)

#pbar.close()
