"""
MVTec AD 데이터셋 인덱싱 유틸리티.

MIMII 파이프라인의 dataset.py는 normal/abnormal 두 폴더를 train/test로
무작위 분할(split_dataset)했지만, MVTec AD는 데이터셋 자체가 이미
    train/good              (정상만, 학습용)
    test/good, test/<defect> (정상+이상, 평가용, ground_truth mask 포함)
로 미리 분리되어 있으므로 여기서는 "분할"이 아니라 "인덱싱"만 수행한다.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from glob import glob
from typing import List, Optional


@dataclass
class TestItem:
    path: str
    label: int              # 0=정상, 1=이상
    defect_type: str
    mask_path: Optional[str]


def _list_images(directory: str) -> List[str]:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif")
    paths: List[str] = []
    for ext in exts:
        paths.extend(glob(os.path.join(directory, ext)))
    return sorted(paths)


def list_train_images(data_root, category: str) -> List[str]:
    """train/good 아래의 정상 이미지 경로 리스트를 반환한다."""
    train_good_dir = os.path.join(str(data_root), category, "train", "good")
    if not os.path.isdir(train_good_dir):
        raise FileNotFoundError(f"학습 폴더를 찾을 수 없습니다: {train_good_dir}")
    return _list_images(train_good_dir)


def list_test_items(data_root, category: str) -> List[TestItem]:
    """test/<defect_type> 아래의 모든 이미지에 라벨과 ground-truth mask 경로를 붙여 반환한다."""
    cat_root = os.path.join(str(data_root), category)
    test_root = os.path.join(cat_root, "test")
    if not os.path.isdir(test_root):
        raise FileNotFoundError(f"테스트 폴더를 찾을 수 없습니다: {test_root}")

    items: List[TestItem] = []
    for defect_type in sorted(os.listdir(test_root)):
        defect_dir = os.path.join(test_root, defect_type)
        if not os.path.isdir(defect_dir):
            continue

        label = 0 if defect_type == "good" else 1
        for img_path in _list_images(defect_dir):
            mask_path = None
            if label == 1:
                fname = os.path.splitext(os.path.basename(img_path))[0]
                candidate = os.path.join(cat_root, "ground_truth", defect_type, f"{fname}_mask.png")
                mask_path = candidate if os.path.isfile(candidate) else None

            items.append(
                TestItem(
                    path=img_path,
                    label=label,
                    defect_type=defect_type,
                    mask_path=mask_path,
                )
            )
    return items


def print_dataset_info(category: str, train_paths: List[str], test_items: List[TestItem]) -> None:
    n_normal_test = sum(1 for item in test_items if item.label == 0)
    n_abnormal_test = sum(1 for item in test_items if item.label == 1)

    print("\n================================")
    print(f"MVTec AD / category = {category}")
    print("================================")
    print(f"Train (good)      : {len(train_paths)}")
    print(f"Test  (good)      : {n_normal_test}")
    print(f"Test  (defect)    : {n_abnormal_test}")

    defect_types = sorted({item.defect_type for item in test_items if item.label == 1})
    if defect_types:
        print(f"Defect types      : {', '.join(defect_types)}")
    print("================================")
