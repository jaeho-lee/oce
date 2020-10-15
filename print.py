import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

model = 'resnet18'
seeds = ['111']
risks = ['avg','cvar_0.2','cvar_0.4','cvar_0.6','cvar_0.8','meanstd_0.5','meanstd_1.0']

riskvectrain = torch.zeros(len(risks),len(seeds),100,7)
riskvectest = torch.zeros(len(risks),len(seeds),100,7)

for i,risk in enumerate(risks):
    for j,seed in enumerate(seeds):
        risklist = torch.load(f'./results/{seed}/{risk}.tsr')
        riskvectrain[i,j] = risklist[0]
        riskvectest[i,j] = risklist[1]

risktrain = riskvectrain.mean(1)
risktest = riskvectest.mean(1)
stdtest = riskvectest.std(1)
stdtrain = riskvectrain.std(1)

# Replace "5" with some integer
# 0: Accuracy
# 1: Loss
# 2: CVaR (alpha = 0.2)
# 3: CVaR (alpha = 0.4)
# 4: CVaR (alpha = 0.6)
# 5: CVaR (alpha = 0.8)

idx = 5

rp1 = risktest[:,:,idx].t()
rp2 = risktrain[:,:,idx].t()
sp1 = stdtest[:,:,idx].t()
sp2 = stdtrain[:,:,idx].t()


pd.DataFrame(torch.cat((rp1,rp2,sp1,sp2),1).numpy()).to_csv('rp.csv')