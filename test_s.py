import argparse
import logging
import os
import torch
import cv2
import numpy as np
from src import utils,eval
from models.model_CLPN_pyr import CLPN
from src.dataset import LoadImages_LOL,LoadImages_ME
from torch.utils.data import DataLoader


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='LOL',
                        help="Specify the directory of the trained model.",
                        dest='model_dir')
    parser.add_argument('--input_dir', '-i', help='Input image directory',
                        dest='input_dir',
                        default='./input')
    parser.add_argument('--device', '-d', default='cuda',
                        help="Device: cuda or cpu.", dest='device')
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default='0',
                        help='CUDA_VISIBLE_DEVICES')
    parser.add_argument('--output_dir',
                        default='./output')
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    if args.device.lower() == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    logging.info(f'Using device {device}')

    net = CLPN()
    net.eval()
    net.cuda()
    total_params = utils.calc_para(net)
    logging.info('Total number of parameters: {}'.format(total_params))
    if args.model_dir == 'ME':
        test = LoadImages_ME(args.input_dir, img_size=None, normalize=False,is_train=False)#(1424,1424)
        model_dir = './checkpoints/ME.pth'
    elif args.model_dir == 'LOL':
        test = LoadImages_LOL(args.input_dir, img_size=None, normalize=False, is_train=False)  # (1424,1424)
        model_dir = './checkpoints/LOL.pth'
    test_loader = DataLoader(test, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    val = open("test_psnr.txt", "a")
    with torch.no_grad():
        ck_name = os.path.splitext(model_dir)[-2]
        checkpoint = torch.load(model_dir,map_location=device)
        logging.info("Current val checkpoint : " + ck_name)
        net.load_state_dict(checkpoint['model'],strict=False)

        psnr_sRGB = 0
        img_num = 0
        for batch in test_loader:
            low_sRGB = batch['low_sRGB']
            normal_sRGB = batch['normal_sRGB']
            low_sRGB_files = batch['low_sRGB_files']

            low_sRGB = low_sRGB.to(device=device, dtype=torch.float32)
            low_sRGB = torch.clamp(low_sRGB, 0, 1)
            normal_sRGB = normal_sRGB.to(device=device, dtype=torch.float32)
            normal_sRGB = torch.clamp(normal_sRGB, 0, 1)

            output,_ = net(low_sRGB)
            output = torch.clamp(output, 0, 1)

            psnr = eval.PSNR(output, normal_sRGB)
            ssim = eval.SSIM(output, normal_sRGB)
            psnr_sRGB = psnr + psnr_sRGB

            output = utils.from_tensor_to_image(output, device=device)

            in_dir, fn = os.path.split(low_sRGB_files[0])
            name, _ = os.path.splitext(fn)
            outsrgb_name = os.path.join(args.output_dir, name + '.png')
            output = output * 255
            cv2.imwrite(outsrgb_name, output.astype(np.uint8))
            print(outsrgb_name)
            img_num += 1
        print(ck_name + '    PSNR : %.6f' % (psnr_sRGB/img_num))
        val.write(ck_name + '    PSNR : %.6f \n' % (psnr_sRGB/img_num))