import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as T
from torch.utils.data import TensorDataset

data_route  = './data/' # EDIT THIS!
cifar_nm    = T.Normalize((0.4914,0.4822,0.4465),(0.247,0.243,0.261))

def get_dataset(dataset):
    if dataset == 'cifar10':
	    tfm_train   = T.Compose([T.RandomCrop(32, padding=4),T.RandomHorizontalFlip(),T.ToTensor(),cifar_nm])
	    tfm_test    = T.Compose([T.ToTensor(),cifar_nm])
	    train_set   = datasets.CIFAR10(data_route,train=True,download=True,transform=tfm_train)
	    test_set    = datasets.CIFAR10(data_route,train=False,download=False,transform=tfm_test)
    else:
        raise ValueError('Unknown dataset')
    return train_set, test_set

def get_loader(dataset,batch_size=100):
    print("Loading "+dataset+" with batch_size "+str(batch_size))
    train_set,test_set  = get_dataset(dataset)
    train_loader    = DataLoader(train_set,batch_size=batch_size,shuffle=True,drop_last=True)
    test_loader     = DataLoader(test_set,batch_size=batch_size,shuffle=False,drop_last=False)
    return train_loader,test_loader