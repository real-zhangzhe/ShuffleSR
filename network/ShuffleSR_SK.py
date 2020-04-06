import torch
import torch.nn as nn


class SKConv(nn.Module):
    def __init__(self, features, M, r, stride=1, L=32):
        """ Constructor
        Args:
            features: input channel dimensionality.
            WH: input spatial dimensionality, used for GAP kernel size.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SKConv, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        wn = lambda x: torch.nn.utils.weight_norm(x)
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                wn(nn.Conv2d(features, features, kernel_size=3 + i * 2, stride=stride, padding=1 + i, groups=features)),
                # wn(nn.Conv2d(features, features, kernel_size=3, dilation=i+1, stride=stride, padding=1 + i, groups=features)),
                nn.ReLU(inplace=False)
            ))
        # self.gap = nn.AvgPool2d(int(WH/stride))
        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Linear(d, features)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            fea = conv(x).unsqueeze_(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        fea_U = torch.sum(feas, dim=1)
        # fea_s = self.gap(fea_U).squeeze_()
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v


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
            nn.ReLU(inplace=True),
            # dw
            # wn(nn.Conv2d(mid_size, mid_size, kernel_size, 1, pad, groups=mid_size//groups_factor, bias=bias)),
            SKConv(features=mid_size, M=2, r=2),
            # pw-linear
            wn(nn.Conv2d(mid_size, out_size, 1, 1, 0, bias=bias)),
            nn.ReLU(inplace=True),
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


class ShuffleSR_SK(nn.Module):
    def __init__(self, n_feats, n_resblocks, expand, scale, wn, groups_factor, bias=False):
        super(ShuffleSR_SK, self).__init__()

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
    model = ShuffleSR_SK(n_feats=28, n_resblocks=12, expand=3, scale=4, wn=wn, groups_factor=1, bias=True).cuda()
    # print(model)

    test_data = torch.rand(1, 3, 48, 48).cuda()
    # test_outputs = model(test_data)
    # print(test_outputs.size())

    from thop import profile
    flops, parameters = profile(model, inputs=(test_data,))
    print('flops = ', int(flops), '\tparameters = ', int(parameters))


# test()
