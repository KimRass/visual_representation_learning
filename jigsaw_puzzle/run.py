import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from kornia.augmentation import RandomCrop, Resize, RandomGrayscale
import random

from jigsaw_puzzle.context_free_network import (
    ContextFreeNetwork
)
from jigsaw_puzzle.permutation_set import (
    get_permutation_set,
    get_permutated_tiles
)
from process_images import (
    load_image,
    show_image,
    save_image
)


def _denormalize_array(img):
    copied_img = img.copy()
    copied_img *= 255.0
    copied_img = np.clip(a=copied_img, a_min=0, a_max=255).astype("uint8")
    return copied_img


def _tensor_to_array(tensor):
    copied_tensor = tensor.clone().squeeze().permute((1, 2, 0)).detach().cpu().numpy()
    copied_tensor = _denormalize_array(copied_tensor)
    return copied_tensor


class SpatiallyJitterColorChannels(nn.Module):
    def __init__(self, shift=1):
        super().__init__()

        self.shift = shift

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        for batch in range(b):
            for ch in range(c):
                x[batch: batch + 1, ch: ch + 1, ...] = torch.roll(
                    x[batch: batch + 1, ch: ch + 1, ...],
                    shifts=(random.randint(-self.shift, self.shift), random.randint(-self.shift, self.shift)),
                    dims=(2, 3)
                )
        return x


if __name__ == "__main__":
    BATCH_SIZE = 1
    RESIZE_SIZE = 256
    CROP_SIZE = 225

    fcn = ContextFreeNetwork()

    img = load_image("examples/ori.jpg")
    # 서로 다른 이미지로 구성된 Batch라고 가정해봅시다.
    image = T.ToTensor()(img).unsqueeze(0).repeat(BATCH_SIZE, 1, 1, 1)
    transform = T.Compose(
        [
            Resize(size=RESIZE_SIZE),
            RandomCrop(size=(CROP_SIZE, CROP_SIZE)),
            SpatiallyJitterColorChannels(shift=2),
            RandomGrayscale(p=0.3)
        ]
    )
    image = transform(image)
    for i in range(BATCH_SIZE):
        # show_image(_tensor_to_array(image[i]))
        save_image(
            img=_tensor_to_array(image[i]),
            path="examples/tranformed.jpg"
        )

    perm_set = get_permutation_set(n_perms=20, n_tiles=9)
    perm_tiles = get_permutated_tiles(image=image, perm_set=perm_set, crop_mode="random")
    for i in range(BATCH_SIZE * 9):
        show_image(_tensor_to_array(perm_tiles[i]))
        save_image(
            img=_tensor_to_array(perm_tiles[i]),
            path=f"""examples/tile{i}.jpg"""
        )

    output = fcn(perm_tiles)
    print(output.shape)
