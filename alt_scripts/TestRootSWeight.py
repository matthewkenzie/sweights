import ROOT as r
from array import array
import numpy as np
from iminuit import Minuit
from scipy.stats import expon

tree = r.TTree("data","data")
tree.ReadFile("data.dat", "mass/D:time/D:mass_fs/D:mass_fb/D", ' ')

c = r.TCanvas("c","c",800,600)
c.Divide(2,2)
c.cd(1)
tree.Draw("mass")
c.cd(2)
tree.Draw("time")
c.cd(3)
tree.Draw("mass_fs:mass")
c.cd(4)
tree.Draw("mass_fb:mass")
c.Update()
c.Modified()
c.Draw()

c.Print('figs/tree.pdf')

import pandas as pd
df = pd.read_pickle("sweights.pkl")
data =  df[['mass','time']].to_numpy()

truth_n_sig = 518  #500
truth_n_bkg = 4982 #5000
truth_n_tot = truth_n_sig + truth_n_bkg

#tsplot = r.TSPlot(1, 1, truth_n_tot, 2, tree)
#tsplot.SetTreeSelection("time:mass:mass_fs:mass_fb")
tsplot = r.TSPlot(0, 1, truth_n_tot, 2, tree)
tsplot.SetTreeSelection("mass:mass_fs:mass_fb")

ne = array('i',[truth_n_sig,truth_n_bkg])
tsplot.SetInitialNumbersOfSpecies(ne)
tsplot.MakeSPlot()
tsplot.FillSWeightsHists(50);

tsigh = tsplot.GetSWeightsHist(0,0)
xminch = tsigh.GetXaxis().GetBinLowEdge(1)
xmaxch = tsigh.GetXaxis().GetBinLowEdge(tsigh.GetNbinsX()+1)
c = r.TCanvas()
tsigh.Draw("LEP")
c.Draw()

weights = np.ndarray( truth_n_tot*2 )
tsplot.GetSWeights(weights)
weights = np.reshape( weights, (-1,2) )

data_w_weights = np.append(data,weights,axis=1)
sorted_data_w_weights = data_w_weights[np.argsort(data_w_weights[:,0])]

print(sorted_data_w_weights)
exit(1)

def expnll(lambd):
    b = expon(0, lambd)
    # normalisation factors are needed for time_pdfs, since x range is restricted
    bn = np.diff(b.cdf(trange))
    return -np.sum( data_w_weights[:,2] * np.log( b.pdf(data_w_weights[:,1]) / bn ) )


mi = Minuit(expnll, lambd=2, limit_lambd=(1,3),
            errordef=Minuit.LIKELIHOOD,
            pedantic=False)

mi.migrad()
mi.hesse()
print( mi.get_param_states() )
fit_lambd_sig_tsplot = mi.values['lambd']
print( fit_lambd_sig_tsplot )


input()
