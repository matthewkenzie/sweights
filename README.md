# sWeight Tests

Provide here an sWeighting class which implements sweights in 6 different ways for a comparison:

 1. Using the ``summation" method from the original sPlot paper
 2. Using the ``integration" method rederived by Schmelling
 3. Using the ``refit" method and taking the covariance matrix of this (also proposed in the sPlot paper)
 4. Using the ``subhess" method, i.e. taking the sub-covariance matrix for the yields
 5. Using the implementation in ROOT's TSPlot
 6. Using the implementation in RooStat's SPlot

 Then have two scripts which runs some tests on a toy example both with low stats and high stats

