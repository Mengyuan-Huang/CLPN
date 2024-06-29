import torch
import torch.nn as nn
from models.condition_net import ConditionNet
from models.LPTN_pyr import LPTN


class CLPN(nn.Module):
    def __init__(self):
        super(CLPN, self).__init__()
        self.condition_net = ConditionNet()
        self.lptn_net = LPTN()

    def forward(self, rgb):
        condition = self.condition_net(rgb)
        out, pyr_out = self.lptn_net(rgb, condition)
        return out, pyr_out


if __name__ == "__main__":
    net = CLPN()
    srgb = torch.randn((1, 3, 256, 256))
    xyz = torch.randn((1, 3, 256, 256))
    xyz_manu = torch.randn((5, 3, 1024, 1024))
    net.cuda()
    out = net(srgb.cuda())

    k = 0
    params = list(net.parameters())
    for i in params:
        l = 1
        for j in i.size():
            l *= j
        k = k + l
    print("总参数数量和：" + str(k))
    print("OK")