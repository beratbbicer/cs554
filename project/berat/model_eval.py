import cv2
import torch
import torch.nn as nn
import numpy as np
from model import FullNetwork
from dataset import get_dataloader

def model_eval(model_path, datapath, savepath):
    batch_size, num_workers = 8,1
    dl_val = get_dataloader(datapath, batch_size,num_workers)
    # print("CUDA Available: ",torch.cuda.is_available())
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FullNetwork().double()#.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    criterion = nn.MSELoss()
    losses = []

    idx = 0
    for _, item in enumerate(dl_val):
        lab256, lab64 = item[1], item[2]
        x = lab64[:,0,:,:].view(-1,1,64,64)#.to(device)
        y256 = lab256[:,1:,:,:].view(-1,2,256,256)#.to(device)
        with torch.no_grad():
            outputs = model(x)
            loss = criterion(outputs, y256)
            losses.append(loss.item())
            for i in range(len(outputs)):
                l,a,b = np.transpose(lab256[i,0,:,:].numpy()), np.transpose(outputs[i,0,:,:].numpy()), np.transpose(outputs[i,1,:,:].numpy())
                bgr = cv2.cvtColor(np.dstack((l,a,b)).astype(np.float32), cv2.COLOR_Lab2BGR)
                cv2.imwrite('{}/{}.jpg'.format(savepath,idx), bgr*255)
                idx += 1
    avg_loss = np.mean(np.array(losses))
    print('val loss: {:.5f}'.format(avg_loss))

model_eval('model.pth', '../../coco/biz/', '../../coco/biz/')
