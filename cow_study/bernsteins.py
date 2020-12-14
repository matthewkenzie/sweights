from scipy.stats import rv_continuous
from scipy.special import binom
from scipy._lib._util import _lazywhere

# B(n,i)(x) = (n!)/(i!*(n-i)!) * (1-x)^(n-i) * x^i

def bernk(n,k,x):
  return binom(n,k) * (1-x)**(n-k) * x**k

def bern(x,*coeffs):
  n = len(coeffs)-1
  res = 0
  for i in range(len(coeffs)):
    res += coeffs[i] * bernk(n,i,x)
  return res

def bernI(x,*coeffs):
  n = len(coeffs)-1
  res = 0
  for i in range(n+1):
    term = 0
    for j in range(i,n+1):
      term += ((-1)**(j-i)) * binom(n,j) * binom(j,i) * (x**(j+1)) / (j+1)
    term *= coeffs[i]
    res += term
  return res

def bernN(*coeffs):
    return 1/bernI(1,*coeffs)

class bpoly1_gen(rv_continuous):
  def _pdf(self,x,p0,p1):
    N = bernN(p0,p1)
    return N * _lazywhere( (x>=0) & (x<=1) , (x,p0,p1), f=lambda x,p0,p1: bern(x,p0,p1), f2=lambda x,p0,p1: 0 )
  def _cdf(self,x,p0,p1):
    N = bernN(p0,p1)
    ret = N * _lazywhere( (x>=0) & (x<=1), (x,p0,p1), f=lambda x,p0,p1: bernI(x,p0,p1), f2=lambda x,p0,p1: 0 )
    ret[x>1] = 1
    return ret
bpoly1 = bpoly1_gen(name='bern1')

class bpoly2_gen(rv_continuous):
  def _pdf(self,x,p0,p1,p2):
    N = bernN(p0,p1,p2)
    return N * _lazywhere( (x>=0) & (x<=1) , (x,p0,p1,p2), f=lambda x,p0,p1,p2: bern(x,p0,p1,p2), f2=lambda x,p0,p1,p2: 0 )
  def _cdf(self,x,p0,p1,p2):
    N = bernN(p0,p1,p2)
    ret = N * _lazywhere( (x>=0) & (x<=1), (x,p0,p1,p2), f=lambda x,p0,p1,p2: bernI(x,p0,p1,p2), f2=lambda x,p0,p1,p2: 0 )
    ret[x>1] = 1
    return ret
bpoly2 = bpoly2_gen(name='bern2')

class bpoly3_gen(rv_continuous):
  def _pdf(self,x,p0,p1,p2,p3):
    N = bernN(p0,p1,p2,p3)
    return N * _lazywhere( (x>=0) & (x<=1) , (x,p0,p1,p2,p3), f=lambda x,p0,p1,p2,p3: bern(x,p0,p1,p2,p3), f2=lambda x,p0,p1,p2,p3: 0 )
  def _cdf(self,x,p0,p1,p2,p3):
    N = bernN(p0,p1,p2,p3)
    ret = N * _lazywhere( (x>=0) & (x<=1), (x,p0,p1,p2,p3), f=lambda x,p0,p1,p2,p3: bernI(x,p0,p1,p2,p3), f2=lambda x,p0,p1,p2,p3: 0 )
    ret[x>1] = 1
    return ret
bpoly3 = bpoly3_gen(name='bern3')

class bpoly4_gen(rv_continuous):
  def _pdf(self,x,p0,p1,p2,p3,p4):
    pars = [p0,p1,p2,p3,p4]
    xpars = [x]+pars
    N = bernN(*pars)
    return N * _lazywhere( (x>=0) & (x<=1) , xpars, f=lambda *xpars: bern(*xpars), f2=lambda *xpars: 0 )
  def _cdf(self,x,p0,p1,p2,p3,p4):
    pars = [p0,p1,p2,p3,p4]
    xpars = [x]+pars
    N = bernN(*pars)
    ret = N * _lazywhere( (x>=0) & (x<=1), xpars, f=lambda *xpars: bernI(*xpars), f2=lambda *xpars: 0 )
    ret[x>1] = 1
    return ret
bpoly4 = bpoly4_gen(name='bern4')

class bpoly5_gen(rv_continuous):
  def _pdf(self,x,p0,p1,p2,p3,p4,p5):
    pars = [p0,p1,p2,p3,p4,p5]
    xpars = [x]+pars
    N = bernN(*pars)
    return N * _lazywhere( (x>=0) & (x<=1) , xpars, f=lambda *xpars: bern(*xpars), f2=lambda *xpars: 0 )
  def _cdf(self,x,p0,p1,p2,p3,p4,p5):
    pars = [p0,p1,p2,p3,p4,p5]
    xpars = [x]+pars
    N = bernN(*pars)
    ret = N * _lazywhere( (x>=0) & (x<=1), xpars, f=lambda *xpars: bernI(*xpars), f2=lambda *xpars: 0 )
    ret[x>1] = 1
    return ret
bpoly5 = bpoly5_gen(name='bern5')

class bpoly6_gen(rv_continuous):
  def _pdf(self,x,p0,p1,p2,p3,p4,p5,p6):
    pars = [p0,p1,p2,p3,p4,p5,p6]
    xpars = [x]+pars
    N = bernN(*pars)
    return N * _lazywhere( (x>=0) & (x<=1) , xpars, f=lambda *xpars: bern(*xpars), f2=lambda *xpars: 0 )
  def _cdf(self,x,p0,p1,p2,p3,p4,p5,p6):
    pars = [p0,p1,p2,p3,p4,p5,p6]
    xpars = [x]+pars
    N = bernN(*pars)
    ret = N * _lazywhere( (x>=0) & (x<=1), xpars, f=lambda *xpars: bernI(*xpars), f2=lambda *xpars: 0 )
    ret[x>1] = 1
    return ret
bpoly6 = bpoly6_gen(name='bern6')


