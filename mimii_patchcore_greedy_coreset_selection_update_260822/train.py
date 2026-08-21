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


import numpy as np
import torch

from tqdm import tqdm

import faiss

faiss.omp_set_num_threads(1)

from config import (
    NORMAL_DIR,
    ABNORMAL_DIR,

    TRAIN_RATIO,
    RANDOM_SEED,

    SAMPLE_RATE,
    DURATION,
    N_FFT,
    HOP_LENGTH,
    N_MELS,
    IMAGE_SIZE,

    CORESET_RATIO,
    CORESET_METHOD,
    CORESET_PROJECTION_DIM,

    PATCH_NEIGHBORHOOD_SIZE,

    MEMORY_ROOT,
)

from dataset import (
    split_dataset,
    print_dataset_info,
)

from audio import (
    wav_to_tensor,
)

from feature_extractor import (
    ResNetFeatureExtractor,
)

from patchcore import (
    aggregate_features,
    CoresetSampler,
    PatchCoreMemory
)


def get_device():

    if torch.backends.mps.is_available():

        return torch.device("mps")

    if torch.cuda.is_available():

        return torch.device("cuda")

    return torch.device("cpu")


def main():

    # ========================================================
    # Device
    # ========================================================

    device = get_device()

    print(
        f"Device: {device}"
    )


    # ========================================================
    # Dataset split
    # ========================================================

    (
        normal_train,
        normal_test,
        abnormal_test,
    ) = split_dataset(

        NORMAL_DIR,

        ABNORMAL_DIR,

        train_ratio=TRAIN_RATIO,

        random_seed=RANDOM_SEED,
    )


    print_dataset_info(
        normal_train,
        normal_test,
        abnormal_test,
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


    for path in tqdm(
        normal_train,

        desc="Training feature extraction",
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
        / "MIMII_memory.npy"
    )


    np.save(
        memory_path,
        memory,
    )


    # --------------------------------------------------------
    # Save split information
    # --------------------------------------------------------

    split_path = (
        MEMORY_ROOT
        / "dataset_split.npz"
    )


    np.savez(

        split_path,

        normal_train=np.array(
            [
                str(p)
                for p in normal_train
            ],
            dtype=object,
        ),

        normal_test=np.array(
            [
                str(p)
                for p in normal_test
            ],
            dtype=object,
        ),

        abnormal_test=np.array(
            [
                str(p)
                for p in abnormal_test
            ],
            dtype=object,
        ),
    )


    print(
        "\nMemory saved:"
    )

    print(
        memory_path
    )

    print(
        "\nDataset split saved:"
    )

    print(
        split_path
    )


if __name__ == "__main__":

    main()