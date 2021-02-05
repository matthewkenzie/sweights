verbose=False

import subprocess
from multiprocessing import Pool

def run_cmd(cmd):
  print('Running:', cmd)
  subprocess.run(cmd, shell=True, check=True, capture_output=not verbose)

def parallel(cmds):
  pool = Pool(processes=4)
  [ pool.map( run_cmd, cmds ) ]

if __name__== '__main__':

  ntoys = 200
  nevs  = 1000

  jobs = [
           #'-e'     ,
           #'-e -c -I 1',
           #'-e -c -I 2',
           #'-e -c -I 3',
           #'-e -c -S 3 -I 3',
           #'-e -c -p 2 -I 1',
           #'-e -c -p 2 -I 2',
           #'-e -c -p 2 -I 3',
           #'-e -c -S 3 -p 2 -I 3',
           #'-e -c -S 3 -p 3 -I 3'
           #'-e -c -S 3 -p 4 -I 3',
           #'-e -c -p 3 -I 1',
           #'-e -c -p 3 -I 2',
           #'-e -c -p 3 -I 3',
           #'-e -c -p 4 -I 1',
           #'-e -c -p 4 -I 2',
           #'-e -c -p 4 -I 3',
           #'-e -c -S 3 -p 4 -I 3',
          ]

  for opts in jobs:
    cmds = [ 'python newcowanal.py -t {0} -n {1} -P {2}'.format(t,nevs,opts) for t in range(ntoys) ]
    parallel(cmds)
