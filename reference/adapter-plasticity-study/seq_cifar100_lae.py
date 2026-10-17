"""
Split CIFAR-100, 10 tasks x 10 classes, exactly as in LAE (config/base/cifar100_order{1,2,3}.yaml,
libml/data/transform.py build_imagenet_transform(norm=False)):
  - train: RandomResizedCrop(224, bilinear) + RandomHorizontalFlip + ToTensor, applied to the 32x32 image;
  - test:  Resize(256, bilinear) + CenterCrop(224) + ToTensor;
  - NO normalization (LAE passes [0, 1] pixels to the ViT);
  - class orders: LAE order 1 (0..99), 2 and 3 (explicit lists below); full training set, no validation split.

Order is chosen with --ap_class_order {1,2,3} (added by the lae-lora model). LAE lists the original class
at each position; Mammoth maps label c -> class_order[c], so class_order = argsort(LAE list).

Registered as `seq-cifar100-lae`.
"""
import hashlib
import os
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode

from datasets.seq_cifar100 import TCIFAR100, MyCIFAR100
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders
from datasets.utils import set_default_from_args
from utils.conf import base_path

FAKE = os.environ.get('AP_FAKE_DATA') == '1'
IMG = 32 if FAKE else 224
RESIZE = 36 if FAKE else 256
BIL = InterpolationMode.BILINEAR

LAE_ORDERS = {
    1: list(range(100)),
    2: [23, 8, 11, 7, 48, 13, 1, 91, 94, 54, 16, 63, 52, 41, 80, 2, 47, 87, 78, 66, 19, 6, 24, 10, 59, 30, 22, 29,
        83, 37, 93, 81, 43, 99, 86, 28, 34, 88, 44, 14, 84, 70, 4, 20, 15, 21, 31, 76, 57, 67, 73, 50, 69, 25, 98,
        46, 96, 0, 72, 35, 58, 92, 3, 95, 56, 90, 26, 40, 55, 89, 75, 71, 60, 42, 9, 82, 39, 18, 77, 68, 32, 79, 12,
        85, 36, 17, 64, 27, 74, 45, 61, 38, 51, 62, 65, 33, 5, 53, 97, 49],
    3: [53, 37, 65, 51, 4, 20, 38, 9, 10, 81, 44, 36, 84, 50, 96, 90, 66, 16, 80, 33, 24, 52, 91, 99, 64, 5, 58, 76,
        39, 79, 23, 94, 30, 73, 25, 47, 31, 45, 19, 87, 42, 68, 95, 21, 7, 67, 46, 82, 11, 6, 41, 86, 88, 70, 18, 78,
        71, 59, 43, 61, 22, 14, 35, 93, 56, 28, 98, 54, 27, 89, 1, 69, 74, 2, 85, 40, 13, 75, 29, 34, 92, 0, 77, 55,
        49, 3, 62, 12, 26, 48, 83, 60, 57, 63, 15, 32, 8, 97, 72, 17],
}
LAE_SEEDS = {1: 1, 2: 2, 3: 3}          # seed_everything in each LAE order config


def order_sha256(order):
    return hashlib.sha256(','.join(map(str, order)).encode()).hexdigest()


def mammoth_class_order(k):
    return np.argsort(np.array(LAE_ORDERS[int(k)]))


def _fake_split(train: bool, transform):
    ds = object.__new__(MyCIFAR100 if train else TCIFAR100)
    n = 6 if train else 3
    g = np.random.default_rng(0 if train else 1)
    ds.data = g.integers(0, 256, (100 * n, 32, 32, 3), dtype=np.uint8)
    ds.targets = list(np.repeat(np.arange(100), n))
    ds.transform, ds.target_transform = transform, None
    if train:
        ds.not_aug_transform = transforms.Compose([transforms.ToTensor()])
    return ds


class SequentialCIFAR100LAE(ContinualDataset):
    NAME = 'seq-cifar100-lae'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 10
    N_TASKS = 10
    N_CLASSES = 100
    SIZE = (IMG, IMG)
    MEAN, STD = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)      # identity: LAE does not normalize

    TRANSFORM = transforms.Compose([
        transforms.RandomResizedCrop(IMG, interpolation=BIL),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    TEST_TRANSFORM = transforms.Compose([
        transforms.Resize(RESIZE, interpolation=BIL),
        transforms.CenterCrop(IMG),
        transforms.ToTensor(),
    ])

    def __init__(self, args):
        k = int(getattr(args, 'ap_class_order', 1))
        args.class_order = mammoth_class_order(k)
        args.permute_classes = True
        super().__init__(args)
        args.class_order = mammoth_class_order(k)      # re-assert after the base class
        args.permute_classes = True

    def get_data_loaders(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        if FAKE:
            train_dataset = _fake_split(True, self.TRANSFORM)
            test_dataset = _fake_split(False, self.TEST_TRANSFORM)
        else:
            train_dataset = MyCIFAR100(base_path() + 'CIFAR100', train=True, download=True, transform=self.TRANSFORM)
            test_dataset = TCIFAR100(base_path() + 'CIFAR100', train=False, download=True, transform=self.TEST_TRANSFORM)
        return store_masked_loaders(train_dataset, test_dataset, self)

    @staticmethod
    def get_transform():
        return transforms.Compose([transforms.ToPILImage(), SequentialCIFAR100LAE.TRANSFORM])

    @set_default_from_args("backbone")
    def get_backbone():
        return "vit-lae"

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        return transforms.Lambda(lambda x: x)

    @staticmethod
    def get_denormalization_transform():
        return transforms.Lambda(lambda x: x)

    @set_default_from_args('n_epochs')
    def get_epochs():
        return 5

    @set_default_from_args('batch_size')
    def get_batch_size():
        return 24


class SequentialCIFAR100LAE20T(SequentialCIFAR100LAE):
    """Same data, transforms and LAE class orders, split into 20 tasks x 5 classes (protocol step 4)."""
    NAME = 'seq-cifar100-lae-20t'
    N_CLASSES_PER_TASK = 5
    N_TASKS = 20

    @staticmethod
    def get_transform():
        return transforms.Compose([transforms.ToPILImage(), SequentialCIFAR100LAE20T.TRANSFORM])

    # Mammoth registers set_default_from_args per dataset name, so the subclass declares them again
    @set_default_from_args("backbone")
    def get_backbone():
        return "vit-lae"

    @set_default_from_args('n_epochs')
    def get_epochs():
        return 5

    @set_default_from_args('batch_size')
    def get_batch_size():
        return 24
