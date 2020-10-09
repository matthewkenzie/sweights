from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('-e','--nevents', type=int, help='Number of events per toy')
parser.add_argument('-m','--mbins'  , default="auto", type=str          , help='Mass binning')
parser.add_argument('-n','--ntoys'  , type=int, help='Number of toys to run')
parser.add_argument('-s','--starttoy', type=int, default=1, help='Starting number of toys')
opts = parser.parse_args()

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
import os
os.system('mkdir -p logs')
for i in range(opts.starttoy, opts.ntoys+opts.starttoy):
  #cmds.append( 'python run.py -a -n {0} -m {1} -s {2} -b > logs/out_n{0}_m{1}_s{2}.log'.format(opts.nevents,opts.mbins,i) )
  cmds.append( 'python run.py -f -w -n {0} -m {1} -s {2} -b > logs/out_n{0}_m{1}_s{2}.log'.format(opts.nevents,opts.mbins,i) )

submit(cmds)

#pbar.close()
