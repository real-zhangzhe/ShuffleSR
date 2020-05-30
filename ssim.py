from torch.utils.data.dataloader import DataLoader
from torch import device, no_grad, load
from torchnet.meter.averagevaluemeter import AverageValueMeter
from torchvision.transforms import ToPILImage, ToTensor
from numpy import asarray
from skimage.metrics import structural_similarity

from data.DIV2K import DIV2K

scale = 4
LR_path = '/path/to/LR'
HR_path = '/path/to/HR'
load_model_path = '/path/to/model'
device = device('cuda:0')

with no_grad():
    net = load(load_model_path).to(device)
    net.eval()
    div2k = DIV2K(LR_path, HR_path, batch=None, patch_size=None, train=False)
    dataloader = DataLoader(div2k, batch_size=1, shuffle=False, num_workers=0)
    to_image = ToPILImage()
    to_tensor = ToTensor()
    meter = AverageValueMeter()
    mean_meter = AverageValueMeter()

    for i, (input, label, _) in enumerate(dataloader):
        input = input.to(device)
        output = net(input).cpu()
        output[output > 1] = 1
        output[output < 0] = 0
        output_image = to_image(output[0])
        img1 = asarray(output_image)
        img2 = asarray(to_image(label[0]))
        ssim = structural_similarity(img1, img2, data_range=255, multichannel=True)
        print('Pic %d, SSIM = %lf' % (i+1, ssim))
        meter.add(ssim)
    mean_meter.add(meter.value()[0])

    print('mean %s = %lf' % ('SSIM', meter.value()[0]))
