import numpy as np
from PIL import Image
from ISR.models import RDN
from ISR.models import RRDN

#../vision_data/DIV2K_train_LR_bicubic_2x
import os
print(os.listdir())
img = Image.open('sample4x2.png')
lr_img = np.array(img)



#rdn = RDN(weights='psnr-small') #psnr-large, psnr-small, noise-cancel, gans
rdn = RRDN(weights='gans')
sr_img = rdn.predict(lr_img)
out = Image.fromarray(sr_img)
out.save('out_sample.png')