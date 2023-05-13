# References
    # https://github.com/clvrai/Representation-Learning-by-Learning-to-Count/blob/master/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torchvision.models import alexnet
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageNet
from itertools import product
import cv2
from PIL import Image
from pathlib import Path
import numpy as np

from object_counting.model import (
    AlexNetFeatureExtractor,
    CountingFeatureExtractor
)
from process_images import (
    load_image,
    _to_pil,
    show_image
)

BATCH_SIZE = 16
IMG_SIZE = 256
CROP_SIZE = 224
TILE_SIZE = CROP_SIZE // 2
GRAY_PROB = 0.67


def batched_image_to_grid(image, n_cols, normalize=False, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    b, _, h, w = image.shape
    assert b % n_cols == 0,\
        "The batch size should be a multiple of `n_cols` argument"
    pad = max(2, int(max(h, w) * 0.04))
    grid = torchvision.utils.make_grid(tensor=image, nrow=n_cols, normalize=False, padding=pad)
    grid = grid.clone().permute((1, 2, 0)).detach().cpu().numpy()

    if normalize:
        grid *= variance
        grid += mean
    grid *= 255.0
    grid = np.clip(a=grid, a_min=0, a_max=255).astype("uint8")

    for k in range(n_cols + 1):
        grid[:, (pad + h) * k: (pad + h) * k + pad, :] = 255
    for k in range(b // n_cols + 1):
        grid[(pad + h) * k: (pad + h) * k + pad, :, :] = 255
    return grid


class ContrastiveLoss(nn.Module):
    def __init__(self, counting_feat_extr, batch_size, crop_size=CROP_SIZE, M=10):
        super().__init__()

        self.counting_feat_extr = counting_feat_extr
        self.crop_size = crop_size
        self.M = M

        self.tile_size = self.crop_size // 2
        self.resize = T.Resize((self.tile_size, self.tile_size), antialias=True)
        # 아래와 같은 방법보다 SimCLR에서처럼 Matrix를 구성하는 게 더 좋을 것 같습니다.
        ids = torch.as_tensor([(i, j) for i, j in product(range(batch_size), range(batch_size)) if i != j])
        self.x_ids, self.y_ids = ids[:, 0], ids[:, 1]

    def forward(self, image):
        tile1 = image[:, :, : self.tile_size, : self.tile_size]
        tile2 = image[:, :, self.tile_size:, : self.tile_size]
        tile3 = image[:, :, : self.tile_size, self.tile_size:]
        tile4 = image[:, :, self.tile_size:, self.tile_size:]
        resized = self.resize(image)

        tile1_feat = self.counting_feat_extr(tile1)
        tile2_feat = self.counting_feat_extr(tile2)
        tile3_feat = self.counting_feat_extr(tile3)
        tile4_feat = self.counting_feat_extr(tile4)
        resized_feat = self.counting_feat_extr(resized)

        summed_feat = (tile1_feat + tile2_feat + tile3_feat + tile4_feat)
        loss1 = F.mse_loss(resized_feat[self.x_ids], summed_feat[self.x_ids], reduction="sum")
        loss2 = max(0, self.M - F.mse_loss(resized_feat[self.y_ids], summed_feat[self.y_ids], reduction="sum"))
        loss = loss1 + loss2
        return loss


class CustomDataset(Dataset):
    def __init__(self, root, transform=None):
        super().__init__()
        self.root = root
        self.transform = transform
        self.img_paths = list(map(str, Path(self.root).glob("*.jpg")))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = _to_pil(load_image(img_path))

        if self.transform is not None:
            image = self.transform(image)
        return image


if __name__ == "__main__":
    transform1 = T.Compose(
        [
            T.ToTensor(),
            T.CenterCrop(IMG_SIZE),
            T.RandomCrop(CROP_SIZE),
            # T.Normalize(
            #     mean=[0.485, 0.456, 0.406],
            #     std=[0.229, 0.224, 0.225]
            # )
        ]
    )
    transform2 = T.RandomGrayscale(GRAY_PROB)
    ds = CustomDataset(
        root="/Users/jongbeomkim/Documents/datasets/VOCdevkit/VOC2012/JPEGImages", transform=transform1
    )
    dl = DataLoader(dataset=ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    for batch, image in enumerate(dl, start=1):
        image = transform2(image)

        # grid = batched_image_to_grid(image=image, n_cols=int(BATCH_SIZE ** 0.5))
        # show_image(grid)

        sample_feat_extr = AlexNetFeatureExtractor()
        counting_feat_extr = CountingFeatureExtractor(feat_extr=sample_feat_extr)
        criterion = ContrastiveLoss(counting_feat_extr=counting_feat_extr, batch_size=BATCH_SIZE)

        print(criterion(image))
        criterion.x_ids
        criterion.y_ids
