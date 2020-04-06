import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, n_feats, kernel_size, wn, act=nn.ReLU(True), res_scale=1):
        super(Block, self).__init__()
        self.res_scale = res_scale
        body = list()
        expand = 6
        linear = 0.8
        body.append(
            wn(nn.Conv2d(n_feats, n_feats*expand, 1, padding=1//2)))
        body.append(act)
        body.append(
            wn(nn.Conv2d(n_feats*expand, int(n_feats*linear), 1, padding=1//2)))
        body.append(
            wn(nn.Conv2d(int(n_feats*linear), n_feats, kernel_size, padding=kernel_size//2)))

        self.body = nn.Sequential(*body)

    def forward(self, x):
        res = self.body(x) * self.res_scale
        res += x
        return res


class WDSR_B(nn.Module):
    def __init__(self, scale, n_resblocks, n_feats, res_scale=1,  n_colors=3):
        super(WDSR_B, self).__init__()
        kernel_size = 3
        act = nn.ReLU(True)
        # wn = lambda x: x
        wn = lambda x: torch.nn.utils.weight_norm(x)

        self.rgb_mean = torch.tensor([0.448456, 0.437496, 0.404528]).unsqueeze(1).unsqueeze(1).unsqueeze(0).cuda()

        # define head module
        head = list()
        head.append(
            wn(nn.Conv2d(n_colors, n_feats, 3, padding=3//2)))

        # define body module
        body = list()
        for i in range(n_resblocks):
            body.append(
                Block(n_feats, kernel_size, wn=wn, res_scale=res_scale, act=act))

        # define tail module
        tail = list()
        out_feats = scale*scale*n_colors
        tail.append(
            wn(nn.Conv2d(n_feats, out_feats, 3, padding=3//2)))
        tail.append(nn.PixelShuffle(scale))

        skip = list()
        skip.append(
            wn(nn.Conv2d(n_colors, out_feats, 5, padding=5//2))
        )
        skip.append(nn.PixelShuffle(scale))

        # make object members
        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*tail)
        self.skip = nn.Sequential(*skip)

    def forward(self, x):
        x = (x - self.rgb_mean) / 0.5
        s = self.skip(x)
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        x += s
        x = x * 0.5 + self.rgb_mean
        return x


def test():
    net = WDSR_B(scale=4, n_resblocks=8, n_feats=32, res_scale=1).cuda()
    x = torch.randn(1, 3, 48, 48).cuda()
    from thop import profile
    flops, parameters = profile(net, inputs=(x,))
    print('flops = ', int(flops), '\tparameters = ', int(parameters))


# test()
