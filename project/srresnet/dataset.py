import os
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset
import torchvision.transforms.functional as FT


class LowDataset(Dataset):
    def __init__(self, ds_name='train', res_scale=4, hr_dim=96):
        data_folder = 'data'
        data_loc = os.path.join(data_folder, ds_name)
        self.images = os.listdir(data_loc)
        self.images = [data_loc  + '/' + s for s in self.images]
        self.hr_dim = hr_dim
        self.res_scale = res_scale
    
    def __getitem__(self, i):
        img = Image.open(self.images[i], mode='r')
        img = img.convert('RGB')
        hr_img = img.resize((self.hr_dim, self.hr_dim), Image.BICUBIC)
        lr_img = hr_img.resize((self.hr_dim // self.res_scale, self.hr_dim // self.res_scale), Image.BICUBIC)
        img.close()
        return FT.to_tensor(lr_img), FT.to_tensor(hr_img)
    

    def __len__(self):
        return len(self.images)