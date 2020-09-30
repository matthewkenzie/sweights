# implements a covariance correction for weighted data fits
import numpy as np
from scipy.misc import derivative

def partial_derivative(pdf, var, point, data):
  args = point[:]
  def wraps(x):
    args[var] = x
    return pdf(*args, data)
  return derivative(wraps,point[var], dx=1e-6)

# you should pass the pdf function
# which must be in the form pdf(*pars,data) e.g. a 1D Gaussian would be pdf(mean,sigma,x)
# then pass the data (should be an appropriate degree numpy array to be passed to your pdf
# then pass the weights (should have same shape as data)
# then pass the fit values and fitted covariance of the nominal fit

def cov_correct(pdf, data, wts, fvals, fcov, verbose=False):

  dim = len(fvals)
  assert(fcov.shape[0]==dim and fcov.shape[1]==dim)

  Djk = np.zeros(fcov.shape)

  prob = pdf(*fvals, data)
  print(prob)
  for j in range(dim):
    derivj = partial_derivative(pdf,j,fvals,data)
    for k in range(dim):
      derivk = partial_derivative(pdf,k,fvals,data)

      Djk[j,k] = np.sum( wts**2 * (derivj*derivk) / prob**2 )

  corr_cov = fcov * Djk * fcov.T

  if verbose:
    print('Covariance correction for weighted events')
    print('  Original covariance:')
    print('\t', str(fcov).replace('\n','\n\t '))
    print('  Corrected covariance:')
    print('\t', str(corr_cov).replace('\n','\n\t '))

  return corr_cov

