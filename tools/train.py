import torch
import torch.nn.functional as F

def train(model,train_loader,test_loader,
          optimizer,target_loss,test_losses,
          num_steps,print_steps=10000):
    
    model.train()
    opt = optimizer(model.parameters())
    device = next(model.parameters()).device
    
    test_losslist = []
    train_losslist = []
    
    current_step = 0
    while True:
        for i, (x,y) in enumerate(train_loader):
            current_step += 1
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            
            yhat = model(x)
            lossvec = F.cross_entropy(yhat,y,reduction='none')
            loss = target_loss(lossvec)
            loss.backward()
            opt.step()
            
            if (current_step%print_steps == 0):
                test_results = test(model,test_loader,test_losses)
                train_results = test(model,train_loader,test_losses)
                print(f'Steps: {current_step}/{num_steps} \t Test acc: {test_results[0]:.2f}', end='\r')
                test_losslist.append(test_results)
                train_losslist.append(train_results)
            if current_step >= num_steps:
                break
        if current_step >= num_steps:
            break
    print(f'Train acc: {train_losslist[-1][0]:.2f}\t Test acc: {test_losslist[-1][0]:.2f}')
    
    return torch.FloatTensor(train_losslist), torch.FloatTensor(test_losslist)


def test(model,loader,test_losses):
    model.eval()
    device = next(model.parameters()).device
    total = len(loader.dataset)
    
    correct = 0
    count = 0
    losslog = torch.zeros(total).to(device)
    for i, (x,y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            yhat = model(x)
            _,pred = yhat.max(1)
        losslog[count:count+len(x)] = F.cross_entropy(yhat,y,reduction='none')
        correct += pred.eq(y).sum().item()
        count += len(x)
    
    losslist = []
    losslist.append(correct/total*100.0)
    for test_loss in test_losses:
        losslist.append(test_loss(losslog))
    
    model.train()
    return losslist
