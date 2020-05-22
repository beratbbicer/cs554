# uncompyle6 version 3.7.0
# Python bytecode 3.7 (3394)
# Decompiled from: Python 3.5.3 (default, May 15 2020, 22:04:06) 
# [GCC 8.3.0]
# Embedded file name: /media/SSD_Main/batu/vis/project/srresnet/utils.py
# Compiled at: 2020-05-15 14:37:11
# Size of source mod 2**32: 6571 bytes
from PIL import Image
import os, json, random
import torchvision.transforms.functional as FT
import torch, math
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
rgb_weights = torch.FloatTensor([65.481, 128.553, 24.966]).to(device)
imagenet_mean = torch.FloatTensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(2)
imagenet_std = torch.FloatTensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(2)
imagenet_mean_cuda = torch.FloatTensor([0.485, 0.456, 0.406]).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
imagenet_std_cuda = torch.FloatTensor([0.229, 0.224, 0.225]).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)

def convert_image(img, source, target):
    """
    Convert an image from a source format to a target format.
    :param img: image
    :param source: source format, one of 'pil' (PIL image), '[0, 1]' or '[-1, 1]' (pixel value ranges)
    :param target: target format, one of 'pil' (PIL image), '[0, 255]', '[0, 1]', '[-1, 1]' (pixel value ranges),
                   'imagenet-norm' (pixel values standardized by imagenet mean and std.),
                   'y-channel' (luminance channel Y in the YCbCr color format, used to calculate PSNR and SSIM)
    :return: converted image
    """
    if not source in {'[0, 1]', '[-1, 1]', 'pil'}:
        raise AssertionError('Cannot convert from source format %s!' % source)
    elif not target in {'imagenet-norm', '[0, 1]', '[-1, 1]', '[0, 255]', 'y-channel', 'pil'}:
        raise AssertionError('Cannot convert to target format %s!' % target)
    else:
        if source == 'pil':
            img = FT.to_tensor(img)
        elif source == '[0, 1]':
            pass
        elif source == '[-1, 1]':
            img = (img + 1.0) / 2.0
        if target == 'pil':
            img = FT.to_pil_image(img)
        elif target == '[0, 255]':
            img = 255.0 * img
        elif target == '[0, 1]':
            pass
        elif target == '[-1, 1]':
            img = 2.0 * img - 1.0
        elif target == 'imagenet-norm':
            if img.ndimension() == 3:
                img = (img - imagenet_mean) / imagenet_std
            elif img.ndimension() == 4:
                img = (img - imagenet_mean_cuda) / imagenet_std_cuda
        elif target == 'y-channel':
            img = torch.matmul(255.0 * img.permute(0, 2, 3, 1)[:, 4:-4, 4:-4, :], rgb_weights) / 255.0 + 16.0
    return img


class ImageTransforms(object):
    """
    Image transformation pipeline.
    """
    __module__ = __name__
    __qualname__ = 'ImageTransforms'

    def __init__(self, split, crop_size, scaling_factor, lr_img_type, hr_img_type):
        """
        :param split: one of 'train' or 'test'
        :param crop_size: crop size of HR images
        :param scaling_factor: LR images will be downsampled from the HR images by this factor
        :param lr_img_type: the target format for the LR image; see convert_image() above for available formats
        :param hr_img_type: the target format for the HR image; see convert_image() above for available formats
        """
        self.split = split.lower()
        self.crop_size = crop_size
        self.scaling_factor = scaling_factor
        self.lr_img_type = lr_img_type
        self.hr_img_type = hr_img_type
        assert self.split in {'train', 'test'}

    def __call__(self, img):
        """
        :param img: a PIL source image from which the HR image will be cropped, and then downsampled to create the LR image
        :return: LR and HR images in the specified format
        """
        if self.split == 'train':
            left = random.randint(1, img.width - self.crop_size)
            top = random.randint(1, img.height - self.crop_size)
            right = left + self.crop_size
            bottom = top + self.crop_size
            hr_img = img.crop((left, top, right, bottom))
        else:
            x_remainder = img.width % self.scaling_factor
            y_remainder = img.height % self.scaling_factor
            left = x_remainder // 2
            top = y_remainder // 2
            right = left + (img.width - x_remainder)
            bottom = top + (img.height - y_remainder)
            hr_img = img.crop((left, top, right, bottom))
        lr_img = hr_img.resize((int(hr_img.width / self.scaling_factor), int(hr_img.height / self.scaling_factor)), Image.BICUBIC)
        if not (hr_img.width == lr_img.width * self.scaling_factor and hr_img.height == lr_img.height * self.scaling_factor):
            raise AssertionError
        lr_img = convert_image(lr_img, source='pil', target=(self.lr_img_type))
        hr_img = convert_image(hr_img, source='pil', target=(self.hr_img_type))
        return (
         lr_img, hr_img)


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.
    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """
    print('\nDECAYING learning rate.')
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor

    print('The new learning rate is %f\n' % (optimizer.param_groups[0]['lr'],))
# okay decompiling utils.cpython-37.pyc
