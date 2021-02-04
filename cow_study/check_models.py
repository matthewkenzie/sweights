from models import effmodel, sigmodel, sigweffmodel, bkgmodel, bkgnfmodel, bkgweffmodel, bkgnfweffmodel

# ranges
mrange = (5000,5600)
trange = (0,10)

# spars
mmu = 5280
msg = 30
tmu = -1

# bpars
mlb = 400
tsg = 2
tlb = 4
slb = 300
smu = 0.3
ssg = 0.8

# effpars
ea  = 0.2
eb  = 0.2
ed  = -0.15

# check the different models
sigmods = [ sigmodel(mrange,trange,mmu,msg,tlb,cache='load') ,
            sigweffmodel(mrange,trange,mmu,msg,tlb,ea,eb,ed,cache='load')
          ]
bkgmods = [ bkgmodel(mrange,trange,mlb,tmu,tsg,cache='load'),
            bkgnfmodel(mrange,trange,mlb,tmu,tsg,slb,smu,ssg,cache='load'),
            bkgweffmodel(mrange,trange,mlb,tmu,tsg,ea,eb,ed,cache='load'),
            bkgnfweffmodel(mrange,trange,mlb,tmu,tsg,slb,smu,ssg,ea,eb,ed,cache='load')
          ]
effmod  = [ effmodel(mrange,trange,ea,eb,ed,cache='load') ]

allmods = sigmods + bkgmods + effmod

from scipy.integrate import quad
# check integrals
for mod in allmods:
  print( mod.name, 'pdfm:', quad(mod.pdfm, *mrange) )
  print( mod.name, 'pdft:', quad(mod.pdft, *trange) )

# plot distributions
import numpy as np
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1,2,figsize=(16,6))

m = np.linspace(*mrange,200)
t = np.linspace(*trange,200)

for mod in allmods:
  ax[0].plot(m, mod.pdfm(m), label=mod.name )
  ax[1].plot(t, mod.pdft(t), label=mod.name )

ax[0].legend()
ax[0].set_xlabel('Mass [MeV]')
ax[0].set_ylabel('Probability Density')

ax[1].legend()
ax[1].set_xlabel('Decay Time [ps]')
ax[1].set_ylabel('Probability Density')

fig.tight_layout()
fig.savefig('plots/allmodels.pdf')
plt.show()
