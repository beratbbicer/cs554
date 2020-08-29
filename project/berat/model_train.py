import cv2
import torch
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
from model import FullNetwork
from dataset import get_dataloader

def model_train(stop_loss_criteria, datapath, model_savepath):
    batch_size, num_workers = 160,8 
    dl_train = get_dataloader(datapath, batch_size, num_workers)
    print("CUDA Available: ",torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FullNetwork().double().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, verbose=True, factor=0.75)
    criterion = nn.MSELoss()
    num_epochs = 1000
    losses = []

    for epoch in range(num_epochs):
        running_loss = []
        batch_idx = 1
        print('train batch ', end=' ')
        for i, item in enumerate(dl_train):
            lab256, lab64 = item[1], item[2]
            x = lab64[:,0,:,:].view(-1,1,64,64).to(device)
            y256 = lab256[:,1:,:,:].view(-1,2,256,256).to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y256)
            running_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            
            if i % 100 == 99:
                print('{}'.format(batch_idx), end=' ')
            batch_idx += 1   

        train_loss = np.mean(np.array(running_loss))
        print('\n[{}/{}] train loss: {:.5f}'.format(epoch + 1, num_epochs, train_loss))
        scheduler.step(train_loss)
        
        if len(losses) < 3:
            losses.append(train_loss)
        else:
            del losses[0]
            losses.append(train_loss)

        if np.mean(np.array(losses)) <= stop_loss_criteria:
            break
    torch.save(model.state_dict(), model_savepath)

# model_train(70, '../../coco/train/', 'model.pth')