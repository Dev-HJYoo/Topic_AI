import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import time
import pandas as pd

from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)
import numpy as np


class Net(nn.Module):
    def __init__(self, weight=256):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, weight, bias=True)
        self.fc2 = nn.Linear(weight, weight, bias=True)
        self.fc3 = nn.Linear(weight, weight, bias=True)
        self.fc4 = nn.Linear(weight, 10, bias=True)

    def forward(self, x):
        x = x.float()
        h1 = F.relu(self.fc1(x.view(-1, 784)))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        out = F.relu(self.fc4(h3))

        return F.log_softmax(out, dim=1)


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    accuracy = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        pred = output.argmax(dim=1, keepdim=True)
        accuracy += pred.eq(target.view_as(pred)).sum().item() / len(data)
        loss.backward()
        optimizer.step()
        total_loss += loss

    if epoch % 2 == 0:
        print('epoch: {}\tloss: {:0.2f}\taccuracy: {:0.3f}'.format(epoch, total_loss, accuracy / (batch_idx + 1)))


def test(model, device, test_loader):
    model.to(device)
    model.eval()
    accuracy = 0
    cur = time.time()
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)

        pred = output.argmax(dim=1, keepdim=True)
        accuracy += pred.eq(target.view_as(pred)).sum().item() / len(data)

    return time.time() - cur, accuracy / (batch_idx + 1)


# training Hyper parameter
batch_size = 128
epochs = 10
lr = 0.01
momentum = 0.9

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#torch.device("cpu")
print('Training device: {}'.format(device))
print("set hyper parameter Done")

# Get loader
transform = transforms.Compose([
                 transforms.ToTensor(),
                 transforms.Normalize((0.1307,), (0.3081,))])

train_loader = torch.utils.data.DataLoader(
  datasets.MNIST('../data', train=True, download=True,
                 transform=transform),
    batch_size = batch_size, shuffle=True)

print("Loader Done")

# csv format
column = ['GPU time', 'GPU Acc', 'CPU time', 'CPU Acc']
index = ['2', '8', '16', '32', '64', '2', '8', '16', '32', '64']
rows =[]

weights = [256, 1024]
for w in weights:
    net = Net(w).to(device)

    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)

    # train process
    print("Training start!\tweight: {}".format(w))

    for epoch in range(0, epochs):
        cur = time.time()
        train(net, device, train_loader, optimizer, epoch)

    torch.save(net, './model_' + str(w) + '.pt')

    # test process
    model = torch.load('./model_' + str(w) + '.pt')
    number = 10

    row = np.zeros((5, 4)).tolist()
    print('weight: {}'.format(w))
    for _ in range(0, number):
        batch_sizes = [2, 8, 16, 32, 64]
        for i, batch in enumerate(batch_sizes):
            test_batch_size = batch

            # print("GPU test process!!")
            device = torch.device("cuda")
            # print("Device: {}".format(device))
            test_loader = torch.utils.data.DataLoader(
                    datasets.MNIST('../data', train=False, download=True,
                             transform=transform),
                batch_size=test_batch_size, shuffle=True)

            gpu_time, gpu_acc = test(model, device, test_loader)
            # print("GPU time: {:0.3f}\tGPU Accuracy: {:0.3f}".format(gpu_time, gpu_acc))

            # print("CPU test process!!")
            device = torch.device("cpu")
            # print("Device: {}".format(device))
            cpu_time, cpu_acc = test(model, device, test_loader)
            # print("CPU time: {:0.3f}\tCPU Accuracy: {:0.3f}".format(cpu_time, cpu_acc))
            # print(end="\n\n")

            for j, data in enumerate([gpu_time, gpu_acc, cpu_time, cpu_acc]):
                row[i][j] += data

            if _ == number - 1:
                row[i][1] /= 10.
                row[i][3] /= 10.
                rows.append(row[i])
                print('batch size: {}'.format(batch))
                print('GPU time: {:0.3f}\tGPU Acc: {:0.3f}'.format(row[i][0], row[i][1]))
                print('CPU time: {:0.3f}\tCPU Acc: {:0.3f}'.format(row[i][2], row[i][3]))

df = pd.DataFrame(rows, index=index, columns=column)
df.to_csv('result.csv')
print('Done!')

