import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet50
from tqdm import tqdm

from datasets import DetectImageFolder
from transforms import Gray2rgb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

transform = transforms.Compose([
    Gray2rgb(),
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = DetectImageFolder("datasets/detect/", transform=transform)

detectloader = DataLoader(dataset, batch_size=8, shuffle=False)

net = resnet50(num_classes=2).to(device)
net.load_state_dict(torch.load("resnet50.pth"))
net.eval()

normal = []
for data in tqdm(detectloader, desc="Detecting on: "):
    with torch.no_grad():
        inputs, path = data[0].to(device), data[2]

        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)

        for i, pre in enumerate(predicted):
            if pre == 0:
                normal.append(path[i])

with open("detect.txt", 'wt') as f:
    f.write('\n'.join(normal))
