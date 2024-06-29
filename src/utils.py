import numpy as np
import torch
import cv2

def from_tensor_to_image(tensor, device='cuda'):
    """ converts tensor to image """
    tensor = torch.squeeze(tensor, dim=0)
    if device == 'cpu':
        image = tensor.data.numpy()
    else:
        image = tensor.cpu().data.numpy()
    # CHW to HWC
    image = image.transpose((1, 2, 0))
    image = from_rgb2bgr(image)
    return image

def from_image_to_tensor(image):
    image = from_bgr2rgb(image)
    image = im2double(image)  # convert to double
    image = np.array(image)
    assert len(image.shape) == 3, ('Input image should be 3 channels colored '
                                   'images')
    # HWC to CHW
    image = image.transpose((2, 0, 1))
    # return torch.unsqueeze(torch.from_numpy(image), dim=0)
    return torch.from_numpy(image)


def from_bgr2rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert from BGR to RGB


def from_rgb2bgr(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # convert from BGR to RGB


def im2double(im):
    """ Returns a double image [0,1] of the uint im. """
    if im[0].dtype == 'uint8':
        max_value = 255
    elif im[0].dtype == 'uint16':
        max_value = 65535
    return im.astype('float') / max_value

def calc_para(net):
    num_params = 0
    total_str = 'The number of parameters for each sub-block:\n'

    for param in net.parameters():
        num_params += param.numel()

# 计算网络各部分参数量
    for body in net.named_children():
        res_params = 0
        res_str = []
        for param in body[1].parameters():
            res_params += param.numel()
        res_str = '[{:s}] parameters: {}\n'.format(body[0], res_params)
        total_str = total_str + res_str
    total_str = total_str + '[total] parameters: {}\n'.format(num_params)
    return total_str