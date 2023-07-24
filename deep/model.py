import torch
import torch.nn as nn

from torchvision import models
from dataloader import get_dataloader
from torch.nn import functional as F


##########################
pretrained_path = None
device = torch.device('cuda:0')
num_classes = 1
learning_rate = 0.05
B = 4096
ref = 0.7

class Net(nn.Module):
  def __init__(self, ):
    super(Net,self).__init__()
    self.conv1 = nn.Linear(96,64)
    self.fc2 = nn.Linear(64,64)
    self.fc3 = nn.Linear(64,1)
    self.flatten = nn.Flatten()
  def forward(self,x):
    x = self.flatten(x)
    # print(x.shape   )
    x = torch.relu(self.conv1(x))
    x = torch.relu(self.fc2(x))
    x = self.fc3(x)
    return x
  

# if pretrained_path:
#     net = torch.open(pretrained_path)
# else:
#     net = models.resnet18(pretrained=True)
#     num_ftrs = net.fc.in_features
#     net.fc = nn.Linear(num_ftrs, num_classes)
#     print(net.conv1)
#     net.conv1 = nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
#     print(net.conv1)
net = Net()
net.to(device)


train_loader = get_dataloader(B, train=True)
test_loader = get_dataloader(B, train=False)

optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                        lr_lambda=lambda epoch: 0.9 ** epoch,
                                        last_epoch=-1,
                                        verbose=False)
loss_fn = nn.L1Loss()
print(net.conv1.weight.dtype)

def train_loop(dataloader, model, loss_fn, optimizer, epoch):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # 예측(prediction)과 손실(loss) 계산
        # print(X.shape, y.shape)
        X, y = torch.tensor(X, device=device, dtype=net.conv1.weight.dtype), torch.tensor(y, device=device, dtype=net.conv1.weight.dtype)
        # if epoch == 0:
        #     y *= 10
        pred = model(X).squeeze()
        loss = torch.sum(torch.pow(pred-y, 2))

        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    size = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = torch.tensor(X, device=device, dtype=net.conv1.weight.dtype), torch.tensor(y, device=device, dtype=net.conv1.weight.dtype)
            pred = model(X)
            test_loss += torch.sum(torch.pow(pred-y, 2))
            for b in range(len(y)):
                if y[b] > ref and pred[b] > ref:
                    correct += 1
                elif y[b] < -ref and pred[b] < -ref:
                    correct += 1
                if abs(pred[b]) > ref:
                    size += 1                

    print(pred[:10])
    print(y[:10])
    test_loss /= num_batches
    if size > 0:
        print(pred[:10])
        print(y[:10])
        correct /= size
        print(f"Prediction: {size} \n Accuracy: {(100*correct):>0.2f}%, Avg loss: {test_loss:>8f} \n")
    return correct


epochs = 10000
best_loss = 0
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    print("lr: ", optimizer.param_groups[0]['lr'])
    for i in range(8):
        train_loop(train_loader, net, loss_fn, optimizer, epoch=t)
        scheduler.step()
    
    score = test_loop(test_loader, net, loss_fn)
    scheduler.step()
    if score > best_loss:
        torch.save(net, 'best_model.pt')
        print("Saved best model w score: ", score)
print("Done!")