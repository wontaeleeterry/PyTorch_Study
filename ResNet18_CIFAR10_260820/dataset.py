import torch

from torchvision import datasets
from torchvision import transforms

from torch.utils.data import DataLoader
from torch.utils.data import random_split

from config import BATCH_SIZE
from config import NUM_WORKERS


CIFAR10_MEAN = (
    0.4914,
    0.4822,
    0.4465
)

CIFAR10_STD = (
    0.2470,
    0.2435,
    0.2616
)


def get_dataloaders():

    train_transform = transforms.Compose([

        transforms.RandomCrop(
            32,
            padding=4
        ),

        transforms.RandomHorizontalFlip(),

        transforms.ToTensor(),

        transforms.Normalize(
            CIFAR10_MEAN,
            CIFAR10_STD
        )
    ])

    test_transform = transforms.Compose([

        transforms.ToTensor(),

        transforms.Normalize(
            CIFAR10_MEAN,
            CIFAR10_STD
        )
    ])

    # 원본 training dataset
    full_train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=train_transform
    )

    # validation용 dataset
    val_dataset_full = datasets.CIFAR10(
        root="./data",
        train=True,
        download=False,
        transform=test_transform
    )

    train_size = 45000
    val_size = 5000

    generator = torch.Generator().manual_seed(42)

    train_dataset, _ = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=generator
    )

    _, val_dataset = random_split(
        val_dataset_full,
        [train_size, val_size],
        generator=generator
    )

    test_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False
    )

    return (
        train_loader,
        val_loader,
        test_loader
    )