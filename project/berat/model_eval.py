import cv2
import torch
import torch.nn as nn
import numpy as np
from model import FullNetwork
from dataset import get_dataloader

def psnr(img1, img2):
    mse = np.mean(np.square(img1-img2))
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))


def model_eval(model_path, datapath, savepath):
    batch_size, num_workers = 16,1
    dl_val = get_dataloader(datapath, batch_size,num_workers)
    # print("CUDA Available: ",torch.cuda.is_available())
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FullNetwork().double()#.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    criterion = nn.MSELoss()
    losses = []
    psnr_val = []
    idx = 0
    for _, item in enumerate(dl_val):
        ref_bgr, lab256, lab64 = item[0].numpy(), item[1], item[2]
        x = lab64[:,0,:,:].view(-1,1,64,64)#.to(device)
        y256 = lab256[:,1:,:,:].view(-1,2,256,256)#.to(device)
        with torch.no_grad():
            outputs = model(x)
            loss = criterion(outputs, y256)
            losses.append(loss.item())
            for i in range(len(outputs)):
                l,a,b = np.transpose(lab256[i,0,:,:].numpy()), np.transpose(outputs[i,0,:,:].numpy()), np.transpose(outputs[i,1,:,:].numpy())
                bgr = cv2.cvtColor(np.dstack((l,a,b)).astype(np.float32), cv2.COLOR_Lab2BGR)
                psnr_val.append(psnr(ref_bgr[i], np.array(bgr*255, dtype = np.uint8)))
                cv2.imwrite('{}/{}.jpg'.format(savepath,idx), bgr*255)
                idx += 1
    avg_loss = np.mean(np.array(losses))
    print('val loss: {:.5f}'.format(avg_loss))
    print('avg_psnr: {:.5f}'.format(np.mean(psnr_val)))

model_eval('model.pth', '../../coco/validation/', '../../coco/validation_outputs/')
