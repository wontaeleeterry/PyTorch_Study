from pathlib import Path

from sklearn.model_selection import train_test_split


def get_wav_files(directory):
    """
    Return sorted wav files.
    """

    directory = Path(directory)

    if not directory.exists():

        raise FileNotFoundError(
            f"Directory not found:\n{directory}"
        )

    files = sorted(
        directory.glob("*.wav")
    )

    if len(files) == 0:

        raise RuntimeError(
            f"No wav files found:\n{directory}"
        )

    return files


def split_dataset(
    normal_dir,
    abnormal_dir,
    train_ratio=0.8,
    random_seed=42,
):
    """
    Split dataset.

    Normal:
        train + test

    Abnormal:
        test only
    """

    normal_files = get_wav_files(
        normal_dir
    )

    abnormal_files = get_wav_files(
        abnormal_dir
    )

    # --------------------------------------------------------
    # Normal -> train / test
    # --------------------------------------------------------

    normal_train, normal_test = (
        train_test_split(
            normal_files,
            train_size=train_ratio,
            random_state=random_seed,
            shuffle=True,
        )
    )

    # --------------------------------------------------------
    # Abnormal -> test only
    # --------------------------------------------------------

    abnormal_test = abnormal_files

    return (
        normal_train,
        normal_test,
        abnormal_test,
    )


def print_dataset_info(
    normal_train,
    normal_test,
    abnormal_test,
):

    print(
        "\n"
        "===================================="
    )

    print(
        "MIMII Dataset"
    )

    print(
        "===================================="
    )

    print(
        f"Normal train   : "
        f"{len(normal_train)}"
    )

    print(
        f"Normal test    : "
        f"{len(normal_test)}"
    )

    print(
        f"Abnormal test  : "
        f"{len(abnormal_test)}"
    )

    print(
        "===================================="
    )

    print()