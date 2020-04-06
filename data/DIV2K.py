from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToTensor
from PIL import Image
import os
from random import uniform, random, choice, randint
from numpy.random import seed, shuffle
from math import floor


class DIV2K(Dataset):
    def __init__(self, LR_path, HR_path, batch, patch_size, train=True):
        super(DIV2K, self).__init__()
        self.train = train
        self.batch = batch
        self.patch_size = patch_size
        LR_tmp = sorted(os.listdir(LR_path))
        HR_tmp = sorted(os.listdir(HR_path))
        self.LR_files, self.HR_files = [], []
        for file in LR_tmp:
            self.LR_files.append(os.path.join(LR_path, file))
        for file in HR_tmp:
            self.HR_files.append(os.path.join(HR_path, file))
        assert len(self.LR_files) == len(self.HR_files)

        self.transform = ToTensor()
        self.degrees = [0, 90, 180, 270]
        self.scale = Image.open(self.HR_files[0]).size[0] // Image.open(self.LR_files[0]).size[0]

        if not train:
            return

        seed_num = randint(0, 9999)
        seed(seed_num)
        shuffle(self.LR_files)
        seed(seed_num)
        shuffle(self.HR_files)

        self.LR_tensor = None
        self.HR_tensor = None

    def __getitem__(self, item):
        if not self.train:
            LR_file = self.LR_files[item]
            HR_file = self.HR_files[item]
            LR_img = Image.open(LR_file)
            HR_img = Image.open(HR_file)
            LR_tensor = self.transform(LR_img)
            HR_tensor = self.transform(HR_img)
            return LR_tensor, HR_tensor, HR_file

        def data_aug(LR, HR):
            if random() < 0.5:
                LR = LR.transpose(Image.FLIP_LEFT_RIGHT)
                HR = HR.transpose(Image.FLIP_LEFT_RIGHT)
            if random() < 0.5:
                LR = LR.transpose(Image.FLIP_TOP_BOTTOM)
                HR = HR.transpose(Image.FLIP_TOP_BOTTOM)
            if random() < 0.5:
                degree = choice(self.degrees)
                if 90 == degree:
                    LR = LR.transpose(Image.ROTATE_90)
                    HR = HR.transpose(Image.ROTATE_90)
                elif 180 == degree:
                    LR = LR.transpose(Image.ROTATE_180)
                    HR = HR.transpose(Image.ROTATE_180)
                elif 270 == degree:
                    LR = LR.transpose(Image.ROTATE_270)
                    HR = HR.transpose(Image.ROTATE_270)
            return LR, HR

        if 0 == item % self.batch:
            LR_file = self.LR_files[item // self.batch]
            HR_file = self.HR_files[item // self.batch]
            LR_img = Image.open(LR_file)
            HR_img = Image.open(HR_file)
            LR_img, HR_img = data_aug(LR_img, HR_img)
            self.LR_tensor = self.transform(LR_img)     # 3, H, W
            self.HR_tensor = self.transform(HR_img)
            self.LR_size = (self.LR_tensor.shape[1], self.LR_tensor.shape[2])   # H, W

        LR_top = floor(uniform(0, self.LR_size[0] - self.patch_size + 1))
        LR_left = floor(uniform(0, self.LR_size[1] - self.patch_size + 1))
        HR_top = LR_top * self.scale
        HR_left = LR_left * self.scale
        return (self.LR_tensor[:, LR_top: LR_top + self.patch_size, LR_left: LR_left + self.patch_size],
                self.HR_tensor[:, HR_top: HR_top + self.patch_size * self.scale,
                HR_left: HR_left + self.patch_size * self.scale])

    def __len__(self):
        if not self.train:
            return len(self.LR_files)
        return len(self.LR_files) * self.batch

    def get_scale(self):
        return self.scale
