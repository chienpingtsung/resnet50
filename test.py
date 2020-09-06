import torch
import torchvision
from torch.utils.data import DataLoader
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

testloader = DataLoader(dataset, batch_size=8, shuffle=False)

net = resnet50(num_classes=2).to(device)
net.load_state_dict(torch.load("resnet50.pth"))
net.eval()

TP = 0
TN = 0
FP = 0
FN = 0
for data in tqdm(testloader, desc="Testing on: "):
    inputs, labels = data[0].to(device), data[1].to(device)

    outputs = net(inputs)
    _, predicted = torch.max(outputs, 1)

    TP += ((predicted == 1) & (labels == 1)).sum()
    TN += ((predicted == 0) & (labels == 0)).sum()
    FP += ((predicted == 1) & (labels == 0)).sum()
    FN += ((predicted == 0) & (labels == 1)).sum()

p = TP / (TP + FP)
r = TP / (TP + FN)
acc = (TP + TN) / (TP + TN + FP + FN)
print("TP: {}".format(TP))
print("TN: {}".format(TN))
print("FP: {}".format(FP))
print("FN: {}".format(FN))
print("p: {}".format(p))
print("r: {}".format(r))
print("acc: {}".format(acc))
