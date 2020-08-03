import ROOT as r

tree = r.TTree('datatree','datatree')
#tree.ReadFile("TestSPlot_toyMC.dat", "Mes/D:F/D:MesSignal/D:MesBackground/D", ' ')
#tree.ReadFile("TestSPlot_toyMC.dat", "Mes/D:dE/D:F/D:MesSignal/D:MesBackground/D:dESignal/D:dEBackground/D:FSignal/D:FBackground/D", ' ')
tree.ReadFile("data.dat", "mass/D:time/D:mass_fs/D:mass_fb/D", ' ')

tree.Print()

splot = r.TSPlot(1,1,5550,2,tree)

#splot.SetTreeSelection("F:Mes:MesSignal:MesBackground")
splot.SetTreeSelection("time:mass:mass_fs:mass_fb")

from array import array
ne = array('i',[500,5000])
splot.SetInitialNumbersOfSpecies(ne)


print(splot.GetNspecies(), splot.GetNevents())

splot.MakeSPlot("VV")
splot.FillSWeightsHists()

tsigh = splot.GetSWeightsHist(0,0)
tbkgh = splot.GetSWeightsHist(0,1)

c = r.TCanvas('c','c',1600,600)
c.Divide(2,1)
c.cd(1)
tsigh.Draw("LEP")
c.cd(2)
tbkgh.Draw("LEP")
c.Update()
c.Modified()
c.Draw()

weights = r.TMatrixD(5500,4)
splot.GetSWeights(weights)
#weights.Print()
#input()
