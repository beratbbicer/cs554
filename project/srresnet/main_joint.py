import time
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from models import JointNet3
from dataset import JointDataset
from utils import *
import matplotlib.pyplot as plt
import numpy as np

# Data parameters
data_folder = 'data'
crop_size = 128  # crop size of target HR images
scaling_factor = 2  # the scaling factor for the generator; the input LR images will be downsampled from the target HR images by this factor
lr_dim = crop_size // scaling_factor

# Model parameters
large_kernel_size = 9  # kernel size of the first and last convolutions which transform the inputs and outputs
small_kernel_size = 3  # kernel size of all convolutions in-between, i.e. those in the residual and subpixel convolutional blocks
n_blocks = 16  # number of residual blocks

# Learning parameters
checkpoint = None #'test.pth.tar'
batch_size = 128  # batch size
start_epoch = 0  # start at this epoch
iterations = 1e6  # number of training iterations
workers = 4
print_freq = 300
lr = 1e-1
grad_clip = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

cudnn.benchmark = True

def train(train_loader, valid_loader, model, criterion, optimizer, epoch, Ltr, Lval):
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    tr_losses = AverageMeter()
    val_losses = AverageMeter()

    start = time.time()
    for i, (lr_imgs, hr_imgs) in enumerate(train_loader):
        data_time.update(time.time() - start)
        
        lr_imgs = lr_imgs[:,0,:,:].view(-1,1,lr_dim,lr_dim).to(device, dtype=torch.float)
        hr_imgs = hr_imgs[:,1:,:,:].view(-1,2,crop_size,crop_size).to(device, dtype=torch.float)

        sr_imgs = model(lr_imgs)  # (N, 3, 96, 96), in [-1, 1]
        loss = criterion(sr_imgs, hr_imgs)  # scalar

        optimizer.zero_grad()
        loss.backward()

        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)


        optimizer.step()
        tr_losses.update(loss.item(), lr_imgs.size(0))
        Ltr.append(loss.item())

        batch_time.update(time.time() - start)
        start = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]----'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})----'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})----'
                  'Loss {tr_loss.val:.4f} ({tr_loss.avg:.4f})'.format(epoch, i, len(train_loader), batch_time=batch_time,
                                                                    data_time=data_time, tr_loss=tr_losses))
            
    if True:
        model.eval();
        with torch.no_grad():
            for i, (lr_imgs, hr_imgs) in enumerate(valid_loader):
                lr_imgs = lr_imgs[:,0,:,:].view(-1,1,lr_dim,lr_dim).to(device, dtype=torch.float)
                hr_imgs = hr_imgs[:,1:,:,:].view(-1,2,crop_size,crop_size).to(device, dtype=torch.float)
                sr_imgs = model(lr_imgs)
                loss = criterion(sr_imgs, hr_imgs)
                Lval.append(loss.item())
                val_losses.update(loss.item(), lr_imgs.size(0))
        print('Epoch: [{0}] === Val Loss {val_loss.val:.4f} ({val_loss.avg:.4f})'.format(epoch, val_loss=val_losses))
    
    del lr_imgs, hr_imgs, sr_imgs
    
    return Ltr, Lval


def main():
    global start_epoch, epoch, checkpoint

    if checkpoint is None:
        model = JointNet3(large_kernel_size=large_kernel_size, small_kernel_size=small_kernel_size, n_blocks=n_blocks, scaling_factor=scaling_factor)
        optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-6)
    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        Ltr = checkpoint['Ltr']
        Lval = checkpoint['Lval']

    model = model.to(device)
    criterion = nn.MSELoss().to(device)


    train_dataset = JointDataset('tr', res_scale=scaling_factor, hr_dim=crop_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_dataset = JointDataset('v', res_scale=scaling_factor, hr_dim=crop_size)
    valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    epochs = int(iterations // len(train_loader) + 1)
    
    Ltr = []
    Lval = []

    
    for epoch in range(start_epoch, epochs):
        Ltr, Lval = train(train_loader=train_loader, valid_loader=valid_loader, model=model, criterion=criterion, optimizer=optimizer, 
              epoch=epoch, Ltr=Ltr, Lval=Lval)
        torch.save({'epoch': epoch, 'model': model, 'optimizer': optimizer, 'Ltr':Ltr, 'Lval':Lval}, str(epoch) + 'j3_128r2.pth.tar')
        #torch.save({'epoch': epoch, 'model': model, 'optimizer': optimizer, 'Ltr':Ltr, 'Lval':Lval}, 'test.pth.tar')
        
        """
        N = len(Ltr)
        x = np.linspace(0, N, N)
        plt.plot(x, Ltr, "-b", label="tr")
        plt.show()

        N = len(Lval)
        x = np.linspace(0, N, N)
        plt.plot(x, Lval, "-r", label="val")
        plt.show()
        """
        
if __name__ == '__main__':
    main()