import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt

class LowLevelFeaturesNetwork(nn.Module):
    def __init__(self):
        super(LowLevelFeaturesNetwork, self).__init__()
        self.layer1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.layer3 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.layer4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.layer5 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        
        self.layer6 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(512)

    def forward(self, x): 
        x = F.relu(self.bn1(self.layer1(x)))
        x = F.relu(self.bn2(self.layer2(x)))
        x = F.relu(self.bn3(self.layer3(x)))
        x = F.relu(self.bn4(self.layer4(x)))
        x = F.relu(self.bn5(self.layer5(x)))
        x = F.relu(self.bn6(self.layer6(x)))
        return x

class GlobalFeaturesNetwork(nn.Module):
    def __init__(self):
        super(GlobalFeaturesNetwork, self).__init__()
        self.layer1 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        
        self.layer2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(512)

        self.layer3 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(512)

        self.layer4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        self.layer5 = nn.Conv2d(512, 1024, kernel_size=7)

        self.layer6 = nn.Conv2d(1024, 512, kernel_size=1)

        self.layer7 = nn.Conv2d(512, 256, kernel_size=1)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.layer1(x)))
        x = F.relu(self.bn2(self.layer2(x)))
        x = F.relu(self.bn3(self.layer3(x)))
        x = F.relu(self.bn4(self.layer4(x)))
        x = F.relu(self.layer5(x))
        x = F.relu(self.layer6(x))
        x = F.relu(self.layer7(x))
        return x

class MidLevelFeaturesNetwork(nn.Module):
    def __init__(self):
        super(MidLevelFeaturesNetwork, self).__init__()
        self.layer1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(512)

        self.layer2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(256)

    def forward(self, x):
        x = F.relu(self.bn1(self.layer1(x)))
        x = F.relu(self.bn2(self.layer2(x)))
        return x

class FusionNetwork(nn.Module):
    def __init__(self):
        super(FusionNetwork, self).__init__()
        self.layer = nn.Conv2d(512, 256, kernel_size=1)

    def forward(self, x):
        x = torch.sigmoid(self.layer(x))
        return x

class ColorizationNetwork(nn.Module):
    def __init__(self):
        super(ColorizationNetwork, self).__init__()
        self.layer1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)

        self.layer2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.layer3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.layer4 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(32)

        self.layer5 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(32)

        self.layer6 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(16)

        self.layer7 = nn.Conv2d(16, 2, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.relu(self.bn1(self.layer1(x)))
        x = F.interpolate(x, size=(56,56))
        x = F.relu(self.bn2(self.layer2(x)))
        x = F.relu(self.bn3(self.layer3(x)))
        x = F.interpolate(x, size=(112,112))
        x = F.relu(self.bn4(self.layer4(x)))
        x = F.relu(self.bn5(self.layer5(x)))
        x = F.interpolate(x, size=(224,224))
        x = F.relu(self.bn6(self.layer6(x)))
        x = torch.tanh(self.layer7(x))
        return x

class FullNetwork(nn.Module):
    def __init__(self):
        super(FullNetwork, self).__init__()
        self.llfn = LowLevelFeaturesNetwork()
        self.gfn = GlobalFeaturesNetwork()
        self.mlfn = MidLevelFeaturesNetwork()
        self.fusion = FusionNetwork()
        self.cn = ColorizationNetwork()

    def forward(self, x):
        x = self.llfn(x)
        y = self.mlfn(x)
        z = self.gfn(x)
        x = self.fusion(torch.cat((y, z.view(-1,256,1,1).repeat(1,1,28,28)), dim=1))
        x = self.cn(x)
        return x

class VisionDataset(Dataset):
    def __init__(self, rootpath):
        self.datapaths = np.array([f for f in glob.glob(rootpath + '/*.jpg')])
        np.random.shuffle(self.datapaths)

    def __len__(self):
        return len(self.datapaths)

    def __getitem__(self, idx):
        filepath = self.datapaths[idx]
        bgr = cv2.resize(cv2.imread(filepath), (224,224), interpolation= cv2.INTER_NEAREST) # uint8
        lab = cv2.cvtColor(np.float32(bgr)/255.0, cv2.COLOR_BGR2LAB) # bgr to [0,1], then to LAB
        return bgr, torch.as_tensor(np.transpose(lab).astype(np.float64))

    
batch_size = 32
num_workers = 4
ds_train = VisionDataset('../coco/data/')
dl_train = DataLoader(ds_train, batch_size, num_workers)
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FullNetwork().double().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.00001)
criterion = nn.MSELoss()
val_losses = []
num_epochs = 250
lr_decay = 0.95

for epoch in range(num_epochs):
    running_loss = []
    batch_idx = 1
    for i, item in enumerate(dl_train):
        lab = item[1].to(device)
        x = lab[:,0,:,:].view(-1,1,224,224)
        y = lab[:,1:,:,:].view(-1,2,224,224)
        optimizer.zero_grad()
        outputs = model(x) * 127.0
        loss = criterion(outputs, y)
        running_loss.append(loss.item())
        loss.backward()
        optimizer.step()
        if i % 10 == 9:
            print('\tbatch {} loss: {:.5f}'.format(batch_idx, loss.item()))
        batch_idx += 1

        if batch_idx % 5 == 5:
            for g in optimizer.param_groups:
                g['lr'] = g['lr'] * lr_decay

            if lr_decay < 0.999:
                lr_decay += 0.0001
        
    epoch_loss = np.mean(np.array(running_loss))
    print('[{}/{}] epoch loss: {:.5f}'.format(epoch + 1, num_epochs, epoch_loss))
    
    if epoch_loss <= 40:
        break
    else:
        running_loss = []
        batch_idx = 1

torch.save(model.state_dict(), 'model.pth')
batch_size = 4
num_workers = 1
ds_val = VisionDataset('../coco/data/')
dl_val = DataLoader(ds_val, batch_size, num_workers)
figurepath = 'figures/'    
idx = 0
for i, item in enumerate(dl_val):
    ref, lab = item[0].numpy(), item[1].to(device) # 2x224x224x3, 2x3x224x224
    l = lab[:,0,:,:].view(-1,1,224,224)
    outputs = (model(l) * 127.0).detach().cpu().numpy()
    l = l.cpu().numpy()

    for j in range(l.shape[0]):
        a,b = np.transpose(outputs[j,0,:,:]), np.transpose(outputs[j,1,:,:])
        l = np.transpose(lab[j,0,:,:].cpu().numpy())
        bgr = cv2.cvtColor(np.dstack((l,a,b)).astype(np.float32), cv2.COLOR_Lab2BGR)
        cv2.imwrite('{}ref{}.jpg'.format(figurepath,idx), ref[j])
        cv2.imwrite('{}out{}.jpg'.format(figurepath,idx), bgr*255) # scale to [0,255]
        idx += 1
