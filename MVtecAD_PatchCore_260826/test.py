# ================================================================
# macOS(Apple Silicon)에서 faiss(OpenMP/libomp)와 torch의 mps
# 백엔드(Accelerate/Metal)가 같은 프로세스에서 동시에 로드될 때
# 스레딩 런타임이 충돌하여 segmentation fault가 발생하는 경우가 있다.
#
# 이를 피하기 위해 torch/faiss를 import하기 "전에" 아래 환경변수를
# 먼저 설정해야 한다. (import 순서가 중요하므로 파일 맨 위에 위치)
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
import pandas as pd
import torch

from tqdm import tqdm

import faiss

# faiss 자체 스레드 풀도 1개로 제한 (MPS와의 스레드 경합 최소화)
faiss.omp_set_num_threads(1)

from config import (
    DATA_ROOT,
    CATEGORY,

    MEMORY_ROOT,
    RESULT_ROOT,

    RESIZE,
    IMAGE_SIZE,

    PATCH_NEIGHBORHOOD_SIZE,
    REWEIGHT_NUM_NEIGHBORS,
    SEGMENTATION_GAUSSIAN_SIGMA,
    LAYERS,
)

from dataset import (
    list_test_items,
)

from image import (
    image_to_tensor,
    mask_to_tensor,
)

from feature_extractor import (
    ResNetFeatureExtractor,
)

from patchcore import (
    aggregate_features,
    PatchCoreMemory,
)


def get_device():

    if torch.backends.mps.is_available():

        return torch.device("mps")

    if torch.cuda.is_available():

        return torch.device("cuda")

    return torch.device("cpu")


def load_memory(category: str):

    memory_path = (
        MEMORY_ROOT
        / f"{category}_memory.npy"
    )

    grid_path = (
        MEMORY_ROOT
        / f"{category}_grid_size.npy"
    )

    if not memory_path.exists():

        raise FileNotFoundError(
            f"Memory bank not found for '{category}'.\n"
            f"Run: python train.py --category {category}"
        )

    memory = np.load(memory_path)
    grid_size = tuple(np.load(grid_path).tolist())

    return memory, grid_size


def calculate_score(
    path,
    extractor,
    patchcore,
    grid_size,
):

    x = image_to_tensor(

        path,

        resize=RESIZE,

        image_size=IMAGE_SIZE,
    )

    features = (
        extractor.extract(x)
    )

    patches, _ = (
        aggregate_features(
            features,
            patch_size=PATCH_NEIGHBORHOOD_SIZE,
            layers=LAYERS,
        )
    )

    patches = (

        patches

        .squeeze(0)

        .cpu()

        .numpy()
    )

    score, patch_scores = (
        patchcore.predict(
            patches
        )
    )

    seg_map = patchcore.segmentation_map(
        patch_scores,
        grid_size=grid_size,
        output_size=(IMAGE_SIZE, IMAGE_SIZE),
    )

    return score, seg_map


def test_one_category(category: str, device: torch.device) -> None:

    # ========================================================
    # Test items (test/good + test/<defect>)
    # ========================================================

    test_items = list_test_items(
        DATA_ROOT,
        category,
    )

    print(
        f"\n[{category}] Test dataset: {len(test_items)} images "
        f"(normal={sum(1 for t in test_items if t.label == 0)}, "
        f"abnormal={sum(1 for t in test_items if t.label == 1)})"
    )

    # ========================================================
    # Load memory
    # ========================================================

    memory, grid_size = load_memory(
        category
    )

    print(
        f"\nMemory bank: {memory.shape}, grid_size={grid_size}"
    )

    patchcore = (
        PatchCoreMemory(
            k=1,
            num_reweight_neighbors=REWEIGHT_NUM_NEIGHBORS,
            gaussian_sigma=SEGMENTATION_GAUSSIAN_SIGMA,
        )
    )

    patchcore.fit(
        memory
    )

    # ========================================================
    # Feature extractor
    # ========================================================

    extractor = (
        ResNetFeatureExtractor(
            device
        )
    )

    results = []
    seg_maps = []
    gt_masks = []

    for item in tqdm(

        test_items,

        desc=f"[{category}] Test",
    ):

        score, seg_map = calculate_score(

            item.path,

            extractor,

            patchcore,

            grid_size,
        )

        mask = mask_to_tensor(
            item.mask_path,
            resize=RESIZE,
            image_size=IMAGE_SIZE,
        ).squeeze().numpy()

        results.append({

            "file": item.path,

            "label": item.label,

            "defect_type": item.defect_type,

            "score": score,
        })

        seg_maps.append(seg_map.astype(np.float32))
        gt_masks.append(mask.astype(np.uint8))

    # ========================================================
    # Save
    # ========================================================

    RESULT_ROOT.mkdir(

        parents=True,

        exist_ok=True,
    )

    df = pd.DataFrame(
        results
    )

    result_path = (
        RESULT_ROOT
        / f"{category}_test_results.csv"
    )

    df.to_csv(

        result_path,

        index=False,
    )

    seg_path = (
        RESULT_ROOT
        / f"{category}_segmentation.npz"
    )

    np.savez_compressed(

        seg_path,

        seg_maps=np.stack(seg_maps, axis=0),

        gt_masks=np.stack(gt_masks, axis=0),
    )

    print(
        "\nResult saved:"
    )

    print(
        result_path
    )

    print(
        seg_path
    )

    print(
        "\n"
        + str(df.head())
    )


def main():

    parser = argparse.ArgumentParser(
        description="PatchCore 평가용 추론 (MVTec AD)"
    )

    parser.add_argument(
        "--category",
        type=str,
        default=CATEGORY,
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="config.CATEGORIES에 정의된 모든 카테고리에 대해 순차 추론",
    )

    args = parser.parse_args()

    device = get_device()

    print(
        f"Device: {device}"
    )

    if args.all:

        from config import CATEGORIES

        for category in CATEGORIES:

            test_one_category(
                category,
                device,
            )

    else:

        test_one_category(
            args.category,
            device,
        )


if __name__ == "__main__":

    main()
