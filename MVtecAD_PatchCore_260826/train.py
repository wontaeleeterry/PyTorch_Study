# ================================================================
# macOS(Apple Silicon)에서 faiss(OpenMP/libomp)와 torch의 mps
# 백엔드(Accelerate/Metal)가 같은 프로세스에서 동시에 로드될 때
# 스레딩 런타임이 충돌하여 segmentation fault가 발생하는 경우가 있다.
# patchcore.py가 faiss를 import하므로, faiss를 직접 쓰지 않는
# train.py에서도 동일한 위험이 있어 test.py와 동일하게 대응한다.
# ================================================================

import os

os.environ.setdefault(
    "KMP_DUPLICATE_LIB_OK",
    "TRUE",
)

os.environ.setdefault(
    "OMP_NUM_THREADS",
    "1",
)


import argparse

import numpy as np
import torch

from tqdm import tqdm

import faiss

faiss.omp_set_num_threads(1)

from config import (
    DATA_ROOT,
    CATEGORY,

    RESIZE,
    IMAGE_SIZE,

    CORESET_RATIO,
    CORESET_METHOD,
    CORESET_PROJECTION_DIM,

    PATCH_NEIGHBORHOOD_SIZE,
    LAYERS,

    RANDOM_SEED,

    MEMORY_ROOT,
)

from dataset import (
    list_train_images,
    list_test_items,
    print_dataset_info,
)

from image import (
    image_to_tensor,
)

from feature_extractor import (
    ResNetFeatureExtractor,
)

from patchcore import (
    aggregate_features,
    CoresetSampler,
)


def get_device():

    if torch.backends.mps.is_available():

        return torch.device("mps")

    if torch.cuda.is_available():

        return torch.device("cuda")

    return torch.device("cpu")


def train_one_category(category: str, device: torch.device) -> None:

    # ========================================================
    # Dataset indexing (MVTec AD는 train/good, test/* 로 이미 분리되어 있음)
    # ========================================================

    train_paths = list_train_images(
        DATA_ROOT,
        category,
    )

    test_items = list_test_items(
        DATA_ROOT,
        category,
    )

    print_dataset_info(
        category,
        train_paths,
        test_items,
    )

    # ========================================================
    # Feature extractor
    # ========================================================

    extractor = (
        ResNetFeatureExtractor(
            device
        )
    )

    # ========================================================
    # Extract normal training features
    # ========================================================

    all_features = []
    grid_size = None

    for path in tqdm(
        train_paths,

        desc=f"[{category}] Training feature extraction",
    ):

        x = image_to_tensor(

            path,

            resize=RESIZE,

            image_size=IMAGE_SIZE,
        )

        features = (
            extractor.extract(x)
        )

        patches, grid_size = (
            aggregate_features(
                features,
                patch_size=PATCH_NEIGHBORHOOD_SIZE,
                layers=LAYERS,
            )
        )

        # [1, N, D]
        patches = (

            patches

            .squeeze(0)

            .cpu()

            .numpy()
        )

        all_features.append(
            patches
        )

    # ========================================================
    # Merge all patches
    # ========================================================

    all_features = (
        np.concatenate(
            all_features,

            axis=0,
        )
    )

    print(
        "\nAll patch features:"
    )

    print(
        all_features.shape
    )

    # ========================================================
    # Coreset
    # ========================================================

    sampler = CoresetSampler(

        ratio=CORESET_RATIO,

        random_seed=RANDOM_SEED,

        method=CORESET_METHOD,

        projection_dim=CORESET_PROJECTION_DIM,

        device=device,
    )

    memory = sampler.sample(
        all_features
    )

    print(
        "\nMemory bank:"
    )

    print(
        memory.shape
    )

    # ========================================================
    # Save
    # ========================================================

    MEMORY_ROOT.mkdir(

        parents=True,

        exist_ok=True,
    )

    memory_path = (
        MEMORY_ROOT
        / f"{category}_memory.npy"
    )

    np.save(
        memory_path,
        memory,
    )

    # grid_size(h, w)는 test 단계에서 patch score -> 2D 세그멘테이션 맵 복원에 필요
    grid_path = (
        MEMORY_ROOT
        / f"{category}_grid_size.npy"
    )

    np.save(
        grid_path,
        np.array(grid_size, dtype=np.int64),
    )

    print(
        "\nMemory saved:"
    )

    print(
        memory_path
    )


def main():

    parser = argparse.ArgumentParser(
        description="PatchCore 학습 (MVTec AD)"
    )

    parser.add_argument(
        "--category",
        type=str,
        default=CATEGORY,
        help="MVTec AD 카테고리 이름 (예: bottle, cable, ...)",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="config.CATEGORIES에 정의된 모든 카테고리에 대해 순차 학습",
    )

    args = parser.parse_args()

    device = get_device()

    print(
        f"Device: {device}"
    )

    if args.all:

        from config import CATEGORIES

        for category in CATEGORIES:

            train_one_category(
                category,
                device,
            )

    else:

        train_one_category(
            args.category,
            device,
        )


if __name__ == "__main__":

    main()
