# A new look at sWeights

The material in this repository is intended to supplement this paper: ``A new look at sWeights", Dembinski, Kenzie, Lagenbruch and Schmelling

We have provided here a few helper modules / classes which you may find useful:

- `SWeighter.py` - this provides a class which implements sWeights in 6 different ways (depending on users desire)
  1. Using the ``summation" method from the original sPlot paper (referred to as Variant B in our paper)
  2. Using the ``integration" method rederived in our paper, originally by Schmelling, (referred to as Variant A in our paper)
  3. Using the ``refit" method, i.e. taking the covariance matrix of a yield only fit (referred to as Variant Ci in our paper)
  4. Using the ``subhess" method, i.e. taking the sub-covariance matrix for the yields (referred to as Variant Cii in our paper)
  5. Using the implementation in ROOT's TSPlot (this we believe should be equivalent to Variant B but is more susceptible to numerical differences)
  6. Using the implementation in RooStat's SPlot (we found this identical to Variant B (``summation") above in all the cases we tried)
  
- `CovarianceCorrect.py` - this provides a function which implements a correction to the covariance matrix based on the fit result. This follows the original work done by Langenbruch in arXiv:1911.01303.

## Examples tried

Each of the subdirectories in this repository contains a specific example that has been presented in our paper. These scripts can be used as the basis for your code if you are wondering how it can be implemented for your analysis.

Have a look at the notebooks in the `notebooks` folder for some visual examples.

