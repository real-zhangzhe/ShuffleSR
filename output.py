from torch.utils.data.dataloader import DataLoader
from torch import device, no_grad, load
from torchvision.transforms import ToPILImage
from data.DIV2K import DIV2K
import os

scale = 2
LR_path = '/path/to/LR'
HR_path = '/path/to/HR'
load_model_path = '/path/to/model'
output_path = '/output/path'

if not os.path.exists(output_path):
    os.makedirs(output_path)

cnt = 0

with no_grad():
    net = load(load_model_path)
    net.eval()
    device = device('cuda:0')
    div2k = DIV2K(LR_path, HR_path, batch=None, patch_size=None, train=False)
    dataloader = DataLoader(div2k, batch_size=1, shuffle=False, num_workers=0)
    to_image = ToPILImage()

    for input, label, _ in dataloader:
        input = input.to(device)
        output = net(input).cpu()
        output[output > 1] = 1
        output[output < 0] = 0
        output_image = to_image(output[0])
        output_image.save(os.path.join(output_path, str(cnt)+'.png'))
        cnt += 1
