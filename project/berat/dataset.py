import torch
from torch.utils.data import Dataset, DataLoader
import glob
import numpy as np
import cv2

class VisionDataset(Dataset):
    def __init__(self, datapath, prepare_data = False):
        self.datapaths = np.array([f for f in glob.glob(datapath + '/*.jpg')])
        np.random.shuffle(self.datapaths)

    def __len__(self):
        return len(self.datapaths)

    def __getitem__(self, idx):
        f = self.datapaths[idx]
        bgr = cv2.resize(cv2.imread(f), (256,256), interpolation= cv2.INTER_NEAREST) # uint8
        lab = torch.as_tensor(np.transpose(cv2.cvtColor(np.float32(bgr)/255.0, cv2.COLOR_BGR2LAB)).astype(np.float64)) # bgr to [0,1], then to LAB
        lab_downscale4 = torch.as_tensor(np.transpose(cv2.cvtColor(np.float32(cv2.resize(bgr, (64,64), interpolation= cv2.INTER_NEAREST))\
            / 255.0, cv2.COLOR_BGR2LAB)).astype(np.float64))
        return torch.as_tensor(bgr), torch.as_tensor(lab), lab_downscale4

def get_dataloader(datapath, batch_size, num_workers):
    return DataLoader(VisionDataset(datapath),batch_size,num_workers)