"""
MVTec AD 이미지 로딩/전처리.

MIMII 파이프라인의 audio.wav_to_tensor(경로 -> 멜스펙트로그램 텐서)에
대응하는 역할을 한다: 이미지 경로를 받아 ImageNet 정규화가 적용된
[1, 3, H, W] 텐서로 변환한다.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from config import IMAGENET_MEAN, IMAGENET_STD


def build_image_transform(resize: int, image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((resize, resize), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_mask_transform(resize: int, image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((resize, resize), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )


def image_to_tensor(path: str, resize: int, image_size: int) -> torch.Tensor:
    """
    Args:
        path: 이미지 파일 경로
        resize: 짧은 변 기준 resize 크기 (논문 §4.1 기본값 256)
        image_size: center crop 이후 최종 입력 크기 (기본값 224)
    Returns:
        [1, 3, image_size, image_size] 정규화된 텐서
    """
    transform = build_image_transform(resize, image_size)
    image = Image.open(path).convert("RGB")
    tensor = transform(image)
    return tensor.unsqueeze(0)


def load_display_image(path: str, resize: int, image_size: int) -> np.ndarray:
    """
    시각화(visualize.py)용 원본 이미지를 로딩한다.
    학습/추론에 쓰이는 image_to_tensor()와 동일한 resize+center crop을 적용하되,
    ImageNet 정규화는 적용하지 않아 사람이 볼 수 있는 RGB 이미지 그대로 반환한다.

    Returns:
        [image_size, image_size, 3] uint8 numpy array
    """
    transform = transforms.Compose(
        [
            transforms.Resize((resize, resize), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size),
        ]
    )
    image = Image.open(path).convert("RGB")
    image = transform(image)
    return np.array(image, dtype=np.uint8)


def mask_to_tensor(path: str | None, resize: int, image_size: int) -> torch.Tensor:
    """
    ground-truth mask를 [1, 1, image_size, image_size] (0/1) 텐서로 변환.
    path가 None이면(=정상 이미지) 전부 0인 마스크를 반환한다.
    """
    if path is None:
        return torch.zeros(1, 1, image_size, image_size)

    transform = build_mask_transform(resize, image_size)
    mask = Image.open(path).convert("L")
    tensor = transform(mask)
    tensor = (tensor > 0.5).float()
    return tensor.unsqueeze(0)
