import torch.nn as nn
import torch.nn.functional as F
import torch

class Lap_Pyramid_Conv(nn.Module):
    def __init__(self, num_high=3):
        super(Lap_Pyramid_Conv, self).__init__()

        self.num_high = num_high
        self.kernel = self.gauss_kernel()

    def gauss_kernel(self, device=torch.device('cuda'), channels=3):
        kernel = torch.tensor([[1., 4., 6., 4., 1],
                               [4., 16., 24., 16., 4.],
                               [6., 24., 36., 24., 6.],
                               [4., 16., 24., 16., 4.],
                               [1., 4., 6., 4., 1.]])
        kernel /= 256.
        kernel = kernel.repeat(channels, 1, 1, 1)
        kernel = kernel.to(device)
        return kernel

    def downsample(self, x):
        return x[:, :, ::2, ::2]

    def upsample(self, x):
        cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
        cc = cc.permute(0, 1, 3, 2)
        cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
        x_up = cc.permute(0, 1, 3, 2)
        return self.conv_gauss(x_up, 4 * self.kernel)

    def conv_gauss(self, img, kernel):
        img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
        out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
        return out

    def pyramid_decom(self, img):
        current = img
        pyr = []
        for _ in range(self.num_high):
            filtered = self.conv_gauss(current, self.kernel)
            down = self.downsample(filtered)
            up = self.upsample(down)
            if up.shape[2] != current.shape[2] or up.shape[3] != current.shape[3]:
                up = nn.functional.interpolate(up, size=(current.shape[2], current.shape[3]))
            diff = current - up
            pyr.append(diff)
            current = down
        pyr.append(current)
        return pyr

    def pyramid_recons(self, pyr):
        image = pyr[-1]
        for level in reversed(pyr[:-1]):
            up = self.upsample(image)
            if up.shape[2] != level.shape[2] or up.shape[3] != level.shape[3]:
                up = nn.functional.interpolate(up, size=(level.shape[2], level.shape[3]))
            image = up + level
        return image

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_features, in_features, 3, padding=1),
            # CALayer(in_features)
        )
    def forward(self, x):
        return x + self.block(x)


class Res_high(nn.Module):
    def __init__(self, in_features):
        super(Res_high, self).__init__()

        self.block = nn.Sequential(
            ResidualBlock(in_features),
            ResidualBlock(in_features),
            ResidualBlock(in_features),
        )
    def forward(self, x):
        return x + self.block(x)


class SFTLayer(nn.Module):
    def __init__(self):
        super(SFTLayer, self).__init__()
        self.SFT_scale_conv = nn.Sequential(
            nn.Conv2d(16, 16, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(16, 32, (3, 3), (1, 1), (1, 1))
        )
        self.SFT_shift_conv = nn.Sequential(
            nn.Conv2d(16, 16, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(16, 32, (3, 3), (1, 1), (1, 1))
        )
    def forward(self, feature, Mask):
        scale = self.SFT_scale_conv(Mask)
        shift = self.SFT_shift_conv(Mask)
        return feature * (scale + 1) + shift

class ResBlock_SFT(nn.Module):
    def __init__(self):
        super(ResBlock_SFT, self).__init__()
        self.sft0 = SFTLayer()
        self.conv0 = nn.Sequential(
            nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1)),
        )
        self.sft1 = SFTLayer()
        self.conv1 = nn.Sequential(
            nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1)),
        )
    def forward(self, feature, mask):
        fea = self.sft0(feature, mask[:,:16,:,:])
        fea = self.conv0(fea)
        fea = self.sft1(fea, mask[:,16:,:,:])
        fea = self.conv1(fea)
        return fea + feature


class Trans_high(nn.Module):
    def __init__(self, num_residual_blocks, num_high=3):
        super(Trans_high, self).__init__()
        self.num_high = num_high

        model = [nn.Conv2d(9, 32, 3, padding=1),
                 nn.LeakyReLU()]
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(32)]
        model += [nn.Conv2d(32, 32, 3, padding=1)]
        self.model = nn.Sequential(*model)

        for i in range(self.num_high):
            trans_mask_block = nn.Sequential(
                Res_high(32),
                nn.Conv2d(32, 16, 1),
                nn.LeakyReLU(),
                nn.Conv2d(16, 3, 1))
            setattr(self, 'trans_mask_block_{}'.format(str(i)), trans_mask_block)

        self.ResBlock_SFT = ResBlock_SFT()
        self.head_conv = nn.Sequential(
            nn.Conv2d(3, 32, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, pyr_original, fake_low):

        pyr_result = []
        mask = self.model(x)

        for i in range(self.num_high):
            mask = nn.functional.interpolate(mask, size=(pyr_original[-2 - i].shape[2], pyr_original[-2 - i].shape[3]))
            input_highfreq = self.head_conv(pyr_original[-2 - i])
            result_highfreq = self.ResBlock_SFT(input_highfreq, mask)
            self.trans_mask_block = getattr(self, 'trans_mask_block_{}'.format(str(i)))
            result_highfreq = self.trans_mask_block(result_highfreq)
            setattr(self, 'result_highfreq_{}'.format(str(i)), result_highfreq)

        for i in reversed(range(self.num_high)):
            result_highfreq = getattr(self, 'result_highfreq_{}'.format(str(i)))
            pyr_result.append(result_highfreq)

        pyr_result.append(fake_low)

        return pyr_result


class CFTLayer(nn.Module):
    def __init__(self):
        super(CFTLayer, self).__init__()
        self.CFT_scale_conv = nn.Sequential(
            nn.Conv2d(32, 32, (1, 1), (1, 1), (0, 0)),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 64, (1, 1), (1, 1), (0, 0))
        )
        self.CFT_shift_conv = nn.Sequential(
            nn.Conv2d(32, 32, (1, 1), (1, 1), (0, 0)),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 64, (1, 1), (1, 1), (0, 0))
        )
    def forward(self, feature, condition):
        scale = self.CFT_scale_conv(condition)
        shift = self.CFT_shift_conv(condition)
        return feature * (scale + 1) + shift


class ResBlock_CFT(nn.Module):
    def __init__(self):
        super(ResBlock_CFT, self).__init__()
        self.cft0 = CFTLayer()
        self.conv0 = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
        )
        self.cft1 = CFTLayer()
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
        )
    def forward(self, feature, condition):
        fea = self.cft0(feature, condition[:, :32, :, :])
        fea = self.conv0(fea)
        fea = self.cft1(fea, condition[:, 32:, :, :])
        fea = self.conv1(fea)
        return fea + feature


class RenderNet_low(nn.Module):
    def __init__(self):
        super(RenderNet_low, self).__init__()
        self.ResBlock_1 = ResBlock_CFT()
        self.ResBlock_2 = ResBlock_CFT()
        self.ResBlock_3 = ResBlock_CFT()
        self.ResBlock_4 = ResBlock_CFT()

        self.head_conv = nn.Sequential(
            nn.Conv2d(3, 64, (1, 1), (1, 1), (0, 0)),
            nn.ReLU(inplace=True)
        )
        self.tail_conv = nn.Sequential(
            nn.Conv2d(64, 3, (1, 1), (1, 1), (0, 0)),
            nn.Sigmoid()
        )
    def forward(self, input, condition):
        fea_head = self.head_conv(input)
        fea_1 = self.ResBlock_1(fea_head, condition[:, :64, :, :])
        fea_2 = self.ResBlock_2(fea_1, condition[:, 64:128, :, :])
        fea_3 = self.ResBlock_3(fea_2, condition[:, 128:192, :, :])
        fea_4 = self.ResBlock_4(fea_3, condition[:, 192:, :, :])
        out = self.tail_conv(fea_4 + fea_head)
        return out


class LPTN(nn.Module):
    def __init__(self, nrb_high=6, num_high=3):
        super(LPTN, self).__init__()
        self.lap_pyramid = Lap_Pyramid_Conv(num_high)
        trans_high = Trans_high(nrb_high, num_high=num_high)
        self.trans_high = trans_high.cuda()
        renderNet_low = RenderNet_low()
        self.renderNet_low = renderNet_low.cuda()

    def forward(self, real_A_full, condition):
        pyr_A = self.lap_pyramid.pyramid_decom(img=real_A_full)
        fake_B_low = self.renderNet_low(pyr_A[-1], condition)
        real_A_up = nn.functional.interpolate(pyr_A[-1], size=(pyr_A[-2].shape[2], pyr_A[-2].shape[3]))
        fake_B_up = nn.functional.interpolate(fake_B_low, size=(pyr_A[-2].shape[2], pyr_A[-2].shape[3]))
        high_with_low = torch.cat([pyr_A[-2], real_A_up, fake_B_up], 1)
        pyr_A_trans = self.trans_high(high_with_low, pyr_A, fake_B_low)
        fake_B_full = self.lap_pyramid.pyramid_recons(pyr_A_trans)
        return fake_B_full, pyr_A_trans
