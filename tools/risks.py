import torch
import torch.nn as nn
import torch.nn.functional as F

def get_risk(risk_name):
    if risk_name == 'avg':
        return Avg
    elif risk_name == 'cvar':
        return CVaR
    elif risk_name == 'meanstd':
        return MeanStd
    else:
        raise NotImplementedError


def Avg(lossvec):
    return lossvec.mean()
    
def CVaR(lossvec,betabar=0.5):
    return lossvec.topk(int(len(lossvec)*betabar))[0].mean()

def MeanStd(lossvec,c=0):
    return lossvec.mean() + c*lossvec.std()
