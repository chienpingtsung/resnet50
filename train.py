import torch
import torchvision
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from torchvision.models import resnet50
from tqdm import tqdm

from transforms import Gray2rgb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

transform = transforms.Compose([
    Gray2rgb(),
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = torchvision.datasets.ImageFolder("datasets/train/", transform=transform)

train_num = int(len(dataset) * 0.8)
test_num = len(dataset) - train_num
trainset, testset = random_split(dataset, [train_num, test_num])

trainloader = DataLoader(trainset, batch_size=8, shuffle=True)
testloader = DataLoader(testset, batch_size=8, shuffle=False)

net = resnet50(num_classes=2).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters())
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30)

min_loss = 1000
min_loss_epoch = 0
for epoch in range(100):
    for data in tqdm(trainloader, desc="Running epoch {:3}: ".format(epoch + 1)):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    test_loss = 0.0
    test_times = 0
    for data in tqdm(testloader, desc="Testing epoch {:3}: ".format(epoch + 1)):
        inputs, labels = data[0].to(device), data[1].to(device)

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        test_times += 1
    test_loss /= test_times
    print("Epoch {:3} test_loss: {}".format(epoch + 1, test_loss))

    if test_loss < min_loss:
        min_loss = test_loss
        min_loss_epoch = epoch
        torch.save(net.state_dict(), "resnet50.pth")

    if epoch - min_loss_epoch > 50:
        break

    scheduler.step()

print("Best at epoch {}, loss is {}.".format(min_loss_epoch, min_loss))
