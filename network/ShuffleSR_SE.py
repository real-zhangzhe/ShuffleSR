import torch
import torch.nn as nn


class Hswish(nn.Module):
    def forward(self, x):
        clip = torch.clamp(x + 3, 0, 6) / 6
        return x * clip


class Hsigmoid(nn.Module):
    def forward(self, x):
        clip = torch.clamp(x + 3, 0, 6) / 6
        return clip


class SELayer(nn.Module):
    def __init__(self, channel, reduction=1):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            Hsigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Block(nn.Module):
    def __init__(self, kernel_size, in_size, mid_size, out_size, wn, groups_factor, bias=False):
        super(Block, self).__init__()

        self.mid_channels = mid_size
        self.ksize = kernel_size
        pad = kernel_size // 2
        self.pad = pad
        in_size //= 2
        out_size = out_size - in_size

        branch_main = [
            # pw
            wn(nn.Conv2d(in_size, mid_size, 1, 1, 0, bias=bias)),
            # nn.ReLU(inplace=True),
            Hswish(),
            # dw
            wn(nn.Conv2d(mid_size, mid_size, kernel_size, 1, pad, groups=mid_size//groups_factor, bias=bias)),
            # SE
            SELayer(mid_size),
            # pw-linear
            wn(nn.Conv2d(mid_size, out_size, 1, 1, 0, bias=bias)),
            # nn.ReLU(inplace=True),
            Hswish(),
        ]
        self.branch_main = nn.Sequential(*branch_main)

    def forward(self, old_x):
        x_proj, x = self.channel_shuffle(old_x)
        x = self.branch_main(x)
        return torch.cat((x_proj, x), 1)

    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.data.size()
        assert (num_channels % 4 == 0)
        x = x.reshape(batchsize * num_channels // 2, 2, height * width)
        x = x.permute(1, 0, 2)
        x = x.reshape(2, -1, num_channels // 2, height, width)
        return x[0], x[1]


class Up_Sample(nn.Module):
    """PixelShuffle"""
    def __init__(self, upscale_factor):
        super(Up_Sample, self).__init__()
        self.ps = nn.PixelShuffle(upscale_factor=upscale_factor)

    def forward(self, x):
        out = self.ps(x)
        return out


class ShuffleSR_SE(nn.Module):
    def __init__(self, n_feats, n_resblocks, expand, scale, wn, groups_factor, bias=False):
        super(ShuffleSR_SE, self).__init__()

        self.head = wn(nn.Conv2d(3, n_feats, kernel_size=3, stride=1, padding=1, bias=bias))
        self.rgb_mean = torch.tensor([0.448456, 0.437496, 0.404528]).unsqueeze(1).unsqueeze(1).unsqueeze(0).cuda()

        kernel_size = 3
        body = list()
        for i in range(n_resblocks):
            body.append(Block(kernel_size, n_feats, n_feats * expand, n_feats, wn=wn,
                              groups_factor=groups_factor, bias=bias))

        tail = list()
        out_feats = scale * scale * 3
        tail.append(wn(nn.Conv2d(n_feats, out_feats, kernel_size=3, stride=1, padding=1, bias=bias)))
        tail.append(Up_Sample(upscale_factor=scale))

        skip = list()
        skip.append(wn(nn.Conv2d(3, out_feats, kernel_size=5, stride=1, padding=2, bias=bias)))
        skip.append(Up_Sample(upscale_factor=scale))

        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*tail)
        self.skip = nn.Sequential(*skip)
        self._initialize_weights()

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

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
    wn = lambda x: torch.nn.utils.weight_norm(x)
    model = ShuffleSR_SE(n_feats=40, n_resblocks=8, expand=2, scale=4, wn=wn, groups_factor=1, bias=True).cuda()
    # print(model)

    test_data = torch.rand(1, 3, 48, 48).cuda()
    # test_outputs = model(test_data)
    # print(test_outputs.size())

    from thop import profile
    flops, parameters = profile(model, inputs=(test_data,))
    print('flops = ', int(flops), '\tparameters = ', int(parameters))


# test()
