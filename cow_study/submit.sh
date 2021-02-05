# factorising no efficiency
python submit.py --sm sig --bm bkg --sN 1000 --bN 1000 -n 500 -m sw
python submit.py --sm sig --bm bkg --sN 1000 --bN 1000 -n 500 -m cow --gs 0 --gb -1 --Im 1
python submit.py --sm sig --bm bkg --sN 1000 --bN 1000 -n 500 -m cow --gs 0 --gb -1 --Im 2

# factorising with efficiency
python submit.py --sm sigweff --bm bkgweff --sN 1000 --bN 1000 -n 500 -e -m sw
python submit.py --sm sigweff --bm bkgweff --sN 1000 --bN 1000 -n 500 -e -m cow --gs 0 --gb -1 --Im 1
python submit.py --sm sigweff --bm bkgweff --sN 1000 --bN 1000 -n 500 -e -m cow --gs 0 --gb -1 --Im 2
python submit.py --sm sigweff --bm bkgweff --sN 1000 --bN 1000 -n 500 -e -m cow --gs 0 --gb -1 --Im 3
python submit.py --sm sigweff --bm bkgweff --sN 1000 --bN 1000 -n 500 -e -m cow --gs 0 --gb 2 --Im 1
python submit.py --sm sigweff --bm bkgweff --sN 1000 --bN 1000 -n 500 -e -m cow --gs 0 --gb 2 --Im 2
python submit.py --sm sigweff --bm bkgweff --sN 1000 --bN 1000 -n 500 -e -m cow --gs 0 --gb 2 --Im 3
python submit.py --sm sigweff --bm bkgweff --sN 1000 --bN 1000 -n 500 -e -m cow --gs 0 --gb 3 --Im 1
python submit.py --sm sigweff --bm bkgweff --sN 1000 --bN 1000 -n 500 -e -m cow --gs 0 --gb 3 --Im 2
python submit.py --sm sigweff --bm bkgweff --sN 1000 --bN 1000 -n 500 -e -m cow --gs 0 --gb 3 --Im 3

# non-factorising no efficiency
python submit.py --sm sig --bm bkgnf --sN 1000 --bN 1000 -n 500 -m sw
python submit.py --sm sig --bm bkgnf --sN 1000 --bN 1000 -n 500 -m cow --gs 0 --gb -1 --Im 1
python submit.py --sm sig --bm bkgnf --sN 1000 --bN 1000 -n 500 -m cow --gs 0 --gb -1 --Im 2
python submit.py --sm sig --bm bkgnf --sN 1000 --bN 1000 -n 500 -m cow --gs 0 --gb -1 --Im 3
python submit.py --sm sig --bm bkgnf --sN 1000 --bN 1000 -n 500 -m cow --gs 0 --gb 2 --Im 1
python submit.py --sm sig --bm bkgnf --sN 1000 --bN 1000 -n 500 -m cow --gs 0 --gb 2 --Im 2
python submit.py --sm sig --bm bkgnf --sN 1000 --bN 1000 -n 500 -m cow --gs 0 --gb 2 --Im 3
python submit.py --sm sig --bm bkgnf --sN 1000 --bN 1000 -n 500 -m cow --gs 0 --gb 3 --Im 1
python submit.py --sm sig --bm bkgnf --sN 1000 --bN 1000 -n 500 -m cow --gs 0 --gb 3 --Im 2
python submit.py --sm sig --bm bkgnf --sN 1000 --bN 1000 -n 500 -m cow --gs 0 --gb 3 --Im 3

# non-factorising with efficiency
python submit.py --sm sigweff --bm bkgnfweff --sN 1000 --bN 1000 -n 500 -e -m sw
python submit.py --sm sigweff --bm bkgnfweff --sN 1000 --bN 1000 -n 500 -e -m cow --gs 0 --gb -1 --Im 1
python submit.py --sm sigweff --bm bkgnfweff --sN 1000 --bN 1000 -n 500 -e -m cow --gs 0 --gb -1 --Im 2
python submit.py --sm sigweff --bm bkgnfweff --sN 1000 --bN 1000 -n 500 -e -m cow --gs 0 --gb -1 --Im 3
python submit.py --sm sigweff --bm bkgnfweff --sN 1000 --bN 1000 -n 500 -e -m cow --gs 0 --gb 2 --Im 1
python submit.py --sm sigweff --bm bkgnfweff --sN 1000 --bN 1000 -n 500 -e -m cow --gs 0 --gb 2 --Im 2
python submit.py --sm sigweff --bm bkgnfweff --sN 1000 --bN 1000 -n 500 -e -m cow --gs 0 --gb 2 --Im 3
python submit.py --sm sigweff --bm bkgnfweff --sN 1000 --bN 1000 -n 500 -e -m cow --gs 0 --gb 3 --Im 1
python submit.py --sm sigweff --bm bkgnfweff --sN 1000 --bN 1000 -n 500 -e -m cow --gs 0 --gb 3 --Im 2
python submit.py --sm sigweff --bm bkgnfweff --sN 1000 --bN 1000 -n 500 -e -m cow --gs 0 --gb 3 --Im 3


