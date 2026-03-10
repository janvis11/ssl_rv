from dataclasses import dataclass
from typing import Callable, Tuple

from PIL import Image
from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class TwoCropsTransform:
    """
    Generate two independently augmented views of the same image.
    Used for SimCLR contrastive learning.
    """

    def __init__(self, base_transform: Callable):
        self.base_transform = base_transform

    def __call__(self, x: Image.Image) -> Tuple:
        return self.base_transform(x), self.base_transform(x)


@dataclass
class TransformConfig:
    image_size: int = 224
    color_jitter_strength: float = 0.5
    blur_prob: float = 0.5
    grayscale_prob: float = 0.2
    horizontal_flip_prob: float = 0.5


def get_simclr_transform(config: TransformConfig = TransformConfig()) -> TwoCropsTransform:
    color_jitter = transforms.ColorJitter(
        brightness=0.8 * config.color_jitter_strength,
        contrast=0.8 * config.color_jitter_strength,
        saturation=0.8 * config.color_jitter_strength,
        hue=0.2 * config.color_jitter_strength,
    )

    base_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=config.image_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(p=config.horizontal_flip_prob),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=config.grayscale_prob),
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return TwoCropsTransform(base_transform)


def get_train_transform(image_size: int = 224) -> Callable:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_eval_transform(image_size: int = 224) -> Callable:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )