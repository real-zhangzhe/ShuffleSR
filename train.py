from torch.utils.data.dataloader import DataLoader
from torch import device, nn, optim, save, load, no_grad
from torchnet.meter.averagevaluemeter import AverageValueMeter
from os.path import join, exists
from os import makedirs
from math import log10
import logging
from datetime import datetime
from torchvision.transforms import ToPILImage
import warnings
import argparse

from data.DIV2K import DIV2K

train_LR_path = '/path/to/train_LR'
train_HR_path = '/path/to/train_HR'
val_LR_path = '/path/to/val_LR'
val_HR_path = '/path/to/val_HR'
device = device('cuda:0')


parser = argparse.ArgumentParser()
parser.add_argument('--network', type=str, help='WDSR or ShuffleSR or ShuffleSR_SE or ShuffleSR_SK')
parser.add_argument('--features', type=int, help='network super-params1: features (channels)')
parser.add_argument('--expand', type=int, help='network super-params2: expand (bottleneck expand)')
parser.add_argument('--blocks', type=int, help='network super-params3: blocks (network depth)')
parser.add_argument('--batch_size', type=int, default=16, help='batch size for training (default: 16)')
parser.add_argument('--groups_factor', type=int, default=1, help='groups factor (default: 1)')
parser.add_argument('--patch_size', type=int, default=48, help='patch size (default:48)')
parser.add_argument('--epochs', type=int, default=200, help='epochs (default:200)')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default:1e-3)')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay (default:0)')
parser.add_argument('--lr_decay_epochs', type=int, default=20, help='lr decay 1/2 per N epochs (default:20)')
parser.add_argument('--print_step', type=int, default=10, help='print step (default:10)')
parser.add_argument('--load_model_path', type=str, default=None, help='load model path (default:None)')
parser.add_argument('--save_path', type=str, default='checkpoint',
                    help='save the best checkpoint to save_path (default:\'./checkpoint\')')
parser.add_argument('--log_path', type=str, default='log', help='save logs to log_path (default:\'./log\')')
args = parser.parse_args()

# path verification
if args.load_model_path:
    assert exists(args.load_model_path), 'load_model_path not exist'
assert exists(train_LR_path), 'train_LR_path not exist'
assert exists(train_HR_path), 'train_HR_path not exist'
assert exists(val_LR_path), 'val_LR_path not exist'
assert exists(val_HR_path), 'val_HR_path not exist'
if not exists(args.save_path):
    makedirs(args.save_path)
if not exists(args.log_path):
    makedirs(args.log_path)

# train super parameters
lr = args.lr
batch = args.batch_size
patch_size = args.patch_size
save_path = args.save_path
epochs = args.epochs
lr_decay_epochs = args.lr_decay_epochs
weight_decay = args.weight_decay
print_step = args.print_step
load_model_path = args.load_model_path

# val super parameters
last_psnr = 0
best_psnr = 0
now_time = datetime.now()
log_path = join(args.log_path, now_time.strftime('%Y-%m-%d %H:%M:%S.log'))
psnr_meter = AverageValueMeter()

# set logger
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler(log_path)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
logger.addHandler(handler)

to_img = ToPILImage()


def train():
    global lr, weight_decay
    blocks = args.blocks
    features = args.features
    expand = args.expand
    groups_factor = args.groups_factor
    bias = True
    wn = lambda x: nn.utils.weight_norm(x)

    scale = DIV2K(train_LR_path, train_HR_path, batch, patch_size).get_scale()
    net = None
    if 'WDSR' == args.network:
        from network.wdsr_b import WDSR_B
        net = WDSR_B(scale=scale, n_resblocks=blocks, n_feats=features).to(device)
    elif 'ShuffleSR' == args.network:
        from network.ShuffleSR import ShuffleSR
        net = ShuffleSR(scale=scale, n_resblocks=blocks, n_feats=features, wn=wn, bias=bias,
                        expand=expand, groups_factor=groups_factor).to(device)
    elif 'ShuffleSR_SE' == args.network:
        from network.ShuffleSR_SE import ShuffleSR_SE
        net = ShuffleSR_SE(scale=scale, n_resblocks=blocks, n_feats=features, wn=wn, bias=bias,
                           expand=expand, groups_factor=groups_factor).to(device)
    elif 'ShuffleSR_SK' == args.network:
        from network.ShuffleSR_SK import ShuffleSR_SK
        net = ShuffleSR_SK(scale=scale, n_resblocks=blocks, n_feats=features, wn=wn, bias=bias,
                           expand=expand, groups_factor=groups_factor).to(device)
    else:
        raise NameError('input networks not implement!')

    logger.warning('Net = %s', type(net).__name__)
    logger.warning('blocks = %d', blocks)
    logger.warning('features = %d', features)
    logger.warning('expand = %d', expand)
    logger.warning('scale = %d', scale)
    logger.warning('groups_factor = %d', groups_factor)
    logger.warning('bias = %s', str(bias))
    logger.warning('lr = %f', lr)
    logger.warning('weight_decay = %f', weight_decay)
    logger.warning('batch = %d', batch)
    logger.warning('patch_size = %d', patch_size)
    logger.warning('save path = %s', save_path)
    logger.warning('lr decay epochs = %d', lr_decay_epochs)

    criterion = nn.L1Loss(reduction='mean').to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    if load_model_path is not None:
        net = load(load_model_path).to(device)

    avg_loss = AverageValueMeter()

    for epoch in range(epochs):
        div2k = DIV2K(train_LR_path, train_HR_path, batch, patch_size)
        dataloader = DataLoader(div2k, batch, shuffle=False, num_workers=0)
        for i, (input, label) in enumerate(dataloader):
            input = input.to(device)
            label = label.to(device)
            output = net(input)
            loss = criterion(output, label)
            avg_loss.add(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if 0 == i % print_step:
                logger.info('epoch %d / %d\tlr = %f\tLoss = %f', epoch, epochs - 1,
                            optimizer.param_groups[0]['lr'], loss.item())
        val(net, epoch)
        if 0 == (epoch + 1) % lr_decay_epochs:
            lr = lr / 2
            weight_decay = weight_decay / 2
            optimizer = optim.Adam(net.parameters(), lr, weight_decay=weight_decay)


def val(net, epoch):
    global last_psnr, best_psnr
    batch = 1
    net.eval()
    div2k = DIV2K(val_LR_path, val_HR_path, None, None, train=False)
    dataloader = DataLoader(div2k, batch, shuffle=False, num_workers=0)

    psnr_meter.reset()
    mse = nn.MSELoss(reduction='mean').to(device)
    with no_grad():
        for input, label, _ in dataloader:
            input = input.to(device)
            label = label.to(device)
            output = net(input)
            output[output < 0] = 0
            output[output > 1] = 1
            output *= 255
            label *= 255
            loss = mse(output, label)
            psnr = 10 * log10(255 * 255 / (loss.item()))
            logger.info('PSNR = %f', psnr)
            psnr_meter.add(psnr)
    cur_psnr = psnr_meter.value()[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if cur_psnr > best_psnr:
            best_psnr = cur_psnr
            save(net, join(save_path, now_time.strftime('%Y-%m-%d %H:%M:%S.pt')))
            logger.info('save done!')
    logger.warning('epoch = %d \tPSNR = %f\tbest PSNR = %f',
                   epoch, cur_psnr, best_psnr)
    if cur_psnr < last_psnr:
        logger.warning('PSNR reduce!')
    last_psnr = cur_psnr
    net.train()


if '__main__' == __name__:
    train()
