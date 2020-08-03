#### Class to get you the sweights ala Michael Schmellings description ####

import ROOT as r
from ROOT import RooFit as rf

import numpy as np
from scipy.integrate import quad

class SchmellWeight():

  ### Need to pass:
  ###    dataset: should be a RooDataSet
  ###    var    : should be a RooRealVar (this is the variable you want to get the sweights from, usually an invariant mass)
  ###    pdfs   : a list of the component pdfs (should be a list of RooAbsPdfs)
  ###    yields : a list of the component yields (should be a list of floats)
  ###    range  : by default it will take the range of the RooRealVar passed var, you can change that here
  ###    precision: decrease for faster computation, increase for better precision

  def __init__(self, dataset, var, pdfs, yields, range=None, precision=10000):

    print('Initialising Schmelling Weighter')
    self.precision = precision
    self.dataset = dataset
    self.var     = var
    if range is not None:
      self.range = range
    else:
      self.range = (var.getMin(), var.getMax())
    self.pdfs    = tuple( self.getpdfasnormcurve(pdf) for pdf in pdfs )
    self.yields  = tuple(yields)
    assert( len(self.pdfs) == len(self.yields) )
    self.alphas = self.solveAlphas()
    print('Computed Schmelling alphas')

  def getpdfasnormcurve(self, pdf):

    # plot the pdf on a RooPlot normalized to unity
    prev_bins = self.var.getBins()
    prev_xmin = self.var.getMin()
    prev_xmax = self.var.getMax()
    self.var.setBins(self.precision)
    self.var.setRange(self.range[0],self.range[1])
    fr = self.var.frame()
    pdf.plotOn(fr, rf.Normalization( self.precision, r.RooAbsReal.NumEvent ), rf.Precision(-1) )
    # get this as a RooCurve
    curve = fr.getCurve()
    curve.SetName( pdf.GetName() )
    self.var.setBins(prev_bins)
    self.var.setRange(prev_xmin, prev_xmax)
    # going to have to renormalise this
    norm = quad( self.evalcurve, self.range[0], self.range[1], args=(curve), full_output=1 )
    x = r.Double(-999)
    y = r.Double(-999)
    for p in range( curve.GetN() ):
      curve.GetPoint(p,x,y)
      curve.SetPoint(p,x,y/norm[0])
    norm = quad( self.evalcurve, self.range[0], self.range[1], args=(curve), full_output=1 )
    assert( abs(norm[0]-1.) < 3*norm[1] )
    return curve

  def evalcurve(self, x, curve):
    return curve.interpolate(x)

  def getWklFunc(self, x, k, l):
    assert(k<len(self.pdfs))
    assert(l<len(self.pdfs))

    pdfk = self.pdfs[k]
    pdfl = self.pdfs[l]

    denom = sum( [ self.yields[i]*self.pdfs[i].interpolate(x) for i in range(len(self.pdfs)) ] )

    return ( pdfk.interpolate(x)*pdfl.interpolate(x) ) / denom

  def getWkl(self, k, l):
    return quad( self.getWklFunc, self.range[0], self.range[1], args=(k,l) )

  def getWklMatrix(self):
    dim = len(self.pdfs)
    res = np.zeros((dim,dim))
    for k in range(dim):
      for l in range(dim):
        Wkl = self.getWkl(k,l) # returns the integral with (value, error)
        res[k][l] = Wkl[0]

    self.WklMatrix = res
    return res

  def solveAlphas(self):
    dim = len(self.pdfs)
    alphas = []

    WklMatrix = self.getWklMatrix()

    for i in range(dim):
      # solves for ith weight
      res = np.zeros(dim)
      res[i] = 1

      sol_alphas = np.linalg.solve(WklMatrix,res)
      alphas.append( sol_alphas )

    return alphas

  def getWeight(self, x, ax):
    alphas = self.alphas[ax]
    dim = len(self.pdfs)

    numer = sum( [alphas[i]*self.pdfs[i].interpolate(x) for i in range(dim) ] )
    denom = sum( [self.yields[i]*self.pdfs[i].interpolate(x) for i in range(dim) ] )
    return numer/denom

  def getWeightedDataTree(self, verbose=False):

    print('Adding Schmell Weights to dataset in form of a TTree')

    tree = r.TTree("tree","tree")

    dim = len(self.pdfs)
    import array
    tree_vars = {}
    obsnames = []
    obsvars = self.dataset.get()
    obsiter = obsvars.createIterator()
    var = obsiter.Next()
    while var:
      obsnames.append( var.GetName() )
      tree_vars[ var.GetName() ] = array.array('f',[-99])
      tree.Branch( var.GetName(), tree_vars[var.GetName()], var.GetName()+"/F" )
      var = obsiter.Next()

    for i in range(dim):
      wtname = '%s_wt'%self.pdfs[i].GetName()
      tree_vars[wtname] = array.array('f',[-99] )
      tree.Branch(wtname, tree_vars[wtname], wtname+'/F')

    weight_sums = [ 0. for i in range(dim) ]

    for ev in range(self.dataset.numEntries()):

      varval = self.dataset.get(ev).getRealValue(self.var.GetName())

      for obsname in obsnames:
        tree_vars[obsname][0] = self.dataset.get(ev).getRealValue(obsname)

      ev_weight_sum = 0.
      for i in range(dim):
        wtname = '%s_wt'%self.pdfs[i].GetName()
        tree_vars[wtname][0] = self.getWeight(varval,i)
        weight_sums[i] += tree_vars[wtname][0]
        ev_weight_sum += tree_vars[wtname][0]

      if abs(ev_weight_sum-1.)>1e-3:
        print('WARNING. Event', ev,'has all weight types summing to a value != 1', ev_weight_sum)

      tree.Fill()

      if (ev%1000==0) and verbose:
        print('Event',ev,'/',self.dataset.numEntries())
        for name, val in tree_vars.items():
          print('  {:20s} : {:8.4g}'.format(name,val[0]))

    for i, yld in enumerate(self.yields):
      print('Yield for pdf',self.pdfs[i].GetName(),'is',yld,'with sum of weights', weight_sums[i])
    return tree
