import os
from PIL import Image

data_folder = 'data'
tr_loc = os.path.join(data_folder, 'train/hr') + '/'
images = os.listdir(os.path.join(data_folder, 'train/hr'))
images = [tr_loc + s for s in images]

for img in images[:1]:
    img = Image.open(img, mode='r')
    img = img.convert('RGB')
    hr_img = img.resize((200, 200), Image.BICUBIC)
    lr_img = hr_img.resize((100, 100), Image.BICUBIC)
    hr_img.show()
    lr_img.show()