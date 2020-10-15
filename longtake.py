import argparse,torch,os,random
import numpy as np
from tools.loaders import get_loader
from tools.train import train
from models import *
from functools import partial
import torch.optim as optim
from tools.risks import get_risk,Avg,CVaR,MeanStd

""" ARGS PARSING """
parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=111, help='random seed')
parser.add_argument('--cuda', type=int, default=-1, help='cuda number')

parser.add_argument('--batch_size',type=int,default=100,help='batch size')
parser.add_argument('--train_steps',type=int,default=60000,help='number of training steps')
parser.add_argument('--print_steps',type=int,default=600,help='how often do you want to log the loss?')

parser.add_argument('--target_risk',type=str,default='avg',help='train target, e.g. avg, cvar, meanstd.')
parser.add_argument('--betabar',type=float,default=0.5,help='alpha for batch-CVaR')
parser.add_argument('--stdmult',type=float,default=1.0,help='lambda for batch-SVP')

args = parser.parse_args()

""" FIX RANDOMNESS """
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

""" SETUP CUDA """
DEVICE = torch.device('cpu') if args.cuda==-1 else torch.device('cuda:'+str(args.cuda))

""" LOAD DATASET """
train_loader, test_loader = get_loader('cifar10',batch_size=args.batch_size)

""" MAKE MODEL """
model = ResNet18().to(DEVICE)
optimizer = partial(optim.AdamW)

""" GET RISK THAT YOU WANT TO TRAIN WITH """
target_risk = get_risk(args.target_risk)
if args.target_risk == 'cvar':
    target_risk = partial(target_risk,betabar=args.betabar)
elif args.target_risk == 'meanstd':
    target_risk = partial(target_risk,c=args.stdmult)

""" THE RISK THAT YOU WANT TO RECORD. """
risklist = [Avg,partial(CVaR,betabar=0.2),partial(CVaR,betabar=0.4),partial(CVaR,betabar=0.6),partial(CVaR,betabar=0.8)]

""" TRAIN """
train_risks, test_risks = train(model,train_loader,test_loader,
                                optimizer,target_risk,risklist,
                                args.train_steps,args.print_steps)

""" SAVE """
RISK_PATH = f'./results/{args.seed}'
if not os.path.exists(RISK_PATH):
    os.makedirs(RISK_PATH)

if args.target_risk == 'meanstd':
    torch.save([train_risks,test_risks],RISK_PATH+f'/{args.target_risk}_{args.stdmult}.tsr')
elif args.target_risk == 'avg':
    torch.save([train_risks,test_risks],RISK_PATH+f'/{args.target_risk}.tsr')
else:
    torch.save([train_risks,test_risks],RISK_PATH+f'/{args.target_risk}_{args.betabar}.tsr')