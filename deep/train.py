import torch
import torch.nn as nn

from torchvision import models
from dataloader import get_dataloader
from torch.nn import functional as F


##########################
pretrained_path = './best_model.pt'
device = torch.device('cuda')
num_classes = 3
learning_rate = 0.005
B = 1024
ref = 0.7


if pretrained_path:
    net = torch.load(pretrained_path, map_location=torch.device('cpu'))
else:
    net = models.resnet18(pretrained=True)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, num_classes)
    print(net.conv1)
    net.conv1 = nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
    print(net.conv1)
# net = Net(num_classes=num_classes)
net.to(device)


train_loader = get_dataloader(B, train=True)
test_loader = get_dataloader(B, train=False)

optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                        lr_lambda=lambda epoch: 0.9** epoch,
                                        last_epoch=-1,
                                        verbose=False)
loss_fn = nn.CrossEntropyLoss()
print(net.conv1.weight.dtype)

def train_loop(dataloader, model, loss_fn, optimizer, epoch):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # 예측(prediction)과 손실(loss) 계산
        # print(X.shape, y.shape)

        X, y = torch.tensor(X, device=device, dtype=net.conv1.weight.dtype), torch.tensor(y, device=device, dtype=torch.int64)
        # if epoch == 0:
        #     y *= 10
        # print(X.shape)
        # exit()
        pred = model(X)#.squeeze()
        # print(y.shape)
        # print(pred.shape)
        # print(pred)
        loss = loss_fn(pred, y)
        # print(loss)
        # exit()
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
            X, y = torch.tensor(X, device=device, dtype=net.conv1.weight.dtype), torch.tensor(y, device=device, dtype=torch.int64)
            pred = model(X)
            # test_loss += torch.sum(torch.pow(pred-y, 2))
            test_loss += loss_fn(pred, y)

            correct += torch.sum(torch.where(torch.argmax(pred, axis=1)==y, 1, 0))      
            size += len(pred)
    test_loss = float(test_loss)/float(num_batches)
    # if size > 0:
    # print(pred[:100])
    # print(y[:100])
    # exit()
    correct = float(correct)/size
    print(f"Prediction: {size} \n Accuracy: {(100*correct):>0.2f}%, Avg loss: {test_loss:>8f} \n")
    return correct


epochs = 10000
best_loss = 0
test_loop(test_loader, net, loss_fn)
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    print("lr: ", optimizer.param_groups[0]['lr'])
    for i in range(8):
        train_loop(train_loader, net, loss_fn, optimizer, epoch=t)
        if t == 0:
            scheduler.step()
    
    scheduler.step()
    
    score = test_loop(test_loader, net, loss_fn)
    scheduler.step()
    if score > best_loss:
        torch.save(net, 'best_model.pt')
        print("Saved best model w score: ", score)
print("Done!")