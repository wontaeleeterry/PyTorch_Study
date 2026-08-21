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


import numpy as np
import pandas as pd
import torch

from tqdm import tqdm

import faiss

# faiss 자체 스레드 풀도 1개로 제한 (MPS와의 스레드 경합 최소화)
faiss.omp_set_num_threads(1)

from config import (
    MEMORY_ROOT,
    RESULT_ROOT,

    SAMPLE_RATE,
    DURATION,
    N_FFT,
    HOP_LENGTH,
    N_MELS,
    IMAGE_SIZE,

    PATCH_NEIGHBORHOOD_SIZE,
    REWEIGHT_NUM_NEIGHBORS,
)

from audio import (
    wav_to_tensor,
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


def load_split():

    split_path = (
        MEMORY_ROOT
        / "dataset_split.npz"
    )

    if not split_path.exists():

        raise FileNotFoundError(
            "Dataset split not found.\n"
            "Run train.py first."
        )

    data = np.load(
        split_path,
        allow_pickle=True,
    )

    normal_test = [
        str(x)
        for x in data[
            "normal_test"
        ]
    ]

    abnormal_test = [
        str(x)
        for x in data[
            "abnormal_test"
        ]
    ]

    return (
        normal_test,
        abnormal_test,
    )


def calculate_score(
    path,
    extractor,
    patchcore,
):

    x = wav_to_tensor(

        path,

        sr=SAMPLE_RATE,

        duration=DURATION,

        n_fft=N_FFT,

        hop_length=HOP_LENGTH,

        n_mels=N_MELS,

        image_size=IMAGE_SIZE,
    )


    features = (
        extractor.extract(x)
    )


    patches = (
        aggregate_features(
            features,
            patch_size=PATCH_NEIGHBORHOOD_SIZE,
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


    return (
        score,
        patch_scores,
    )


def main():

    device = get_device()

    print(
        f"Device: {device}"
    )


    # ========================================================
    # Load split
    # ========================================================

    (
        normal_test,
        abnormal_test,
    ) = load_split()


    print(
        "\nTest dataset:"
    )

    print(
        f"Normal   : "
        f"{len(normal_test)}"
    )

    print(
        f"Abnormal : "
        f"{len(abnormal_test)}"
    )


    # ========================================================
    # Load memory
    # ========================================================

    memory_path = (
        MEMORY_ROOT
        / "MIMII_memory.npy"
    )


    if not memory_path.exists():

        raise FileNotFoundError(
            "Memory bank not found.\n"
            "Run train.py first."
        )


    memory = np.load(
        memory_path
    )


    print(
        f"\nMemory bank: "
        f"{memory.shape}"
    )


    patchcore = (
        PatchCoreMemory(
            k=1,
            num_reweight_neighbors=REWEIGHT_NUM_NEIGHBORS,
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


    # ========================================================
    # Normal test
    # ========================================================

    for path in tqdm(

        normal_test,

        desc="Normal test",
    ):

        score, _ = calculate_score(

            path,

            extractor,

            patchcore,
        )


        results.append({

            "file": path,

            "label": 0,

            "score": score,
        })


    # ========================================================
    # Abnormal test
    # ========================================================

    for path in tqdm(

        abnormal_test,

        desc="Abnormal test",
    ):

        score, _ = calculate_score(

            path,

            extractor,

            patchcore,
        )


        results.append({

            "file": path,

            "label": 1,

            "score": score,
        })


    # ========================================================
    # Save
    # ========================================================

    df = pd.DataFrame(
        results
    )


    RESULT_ROOT.mkdir(

        parents=True,

        exist_ok=True,
    )


    result_path = (
        RESULT_ROOT
        / "Test_results.csv"
    )


    df.to_csv(

        result_path,

        index=False,
    )


    print(
        "\nResult saved:"
    )

    print(
        result_path
    )


    print(
        "\n"
        + str(df.head())
    )


if __name__ == "__main__":

    main()