import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as FT


class LowDataset(Dataset):
    def __init__(self):
        data_folder = 'data'
        tr_loc = os.path.join(data_folder, 'train/') + '/'
        self.images = os.listdir(os.path.join(data_folder, 'train/'))
        self.images = [tr_loc + s for s in self.images]
    
    def __getitem__(self, i):
        img = Image.open(self.images[i], mode='r')
        img = img.convert('RGB')
        hr_img = img.resize((200, 200), Image.BICUBIC)
        lr_img = hr_img.resize((100, 100), Image.BICUBIC)
        return FT.to_tensor(lr_img), FT.to_tensor(hr_img)

    def __len__(self):
        return len(self.images)