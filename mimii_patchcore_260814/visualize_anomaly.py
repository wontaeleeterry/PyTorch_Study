import numpy as np
import matplotlib.pyplot as plt

import librosa
import librosa.display

import torch
import torch.nn.functional as F

from scipy.ndimage import zoom
from pathlib import Path


# ============================================================
# Project modules
# ============================================================

from config import (
    MEMORY_ROOT,
    SAMPLE_RATE,
    DURATION,
    N_FFT,
    HOP_LENGTH,
    N_MELS,
    IMAGE_SIZE,
)

from audio import (
    wav_to_tensor,
)

from feature_extractor import (
    ResNetFeatureExtractor,
)

from patchcore import (
    PatchCoreMemory,
)


# ============================================================
# Device
# ============================================================

def get_device():

    if torch.backends.mps.is_available():

        return torch.device("mps")

    elif torch.cuda.is_available():

        return torch.device("cuda")

    else:

        return torch.device("cpu")


# ============================================================
# Load test split
# ============================================================

def load_test_data():

    split_path = (
        MEMORY_ROOT
        / "dataset_split.npz"
    )

    if not split_path.exists():

        raise FileNotFoundError(
            f"Test split not found:\n"
            f"{split_path}"
        )

    data = np.load(
        split_path,
        allow_pickle=True
    )

    normal_test = [
        str(x)
        for x in data["normal_test"]
    ]

    abnormal_test = [
        str(x)
        for x in data["abnormal_test"]
    ]

    return (
        normal_test,
        abnormal_test,
    )


# ============================================================
# Load WAV
# ============================================================

def load_audio(path):

    y, sr = librosa.load(

        path,

        sr=SAMPLE_RATE,

        mono=True,
    )

    return (
        y,
        sr,
    )


# ============================================================
# Create Mel Spectrogram
# ============================================================

def create_mel_spectrogram(
    y,
    sr
):

    mel = librosa.feature.melspectrogram(

        y=y,

        sr=sr,

        n_fft=N_FFT,

        hop_length=HOP_LENGTH,

        n_mels=N_MELS,

        power=2.0,
    )

    mel_db = librosa.power_to_db(

        mel,

        ref=np.max,
    )

    return mel_db


# ============================================================
# Aggregate ResNet feature maps
# ============================================================

def aggregate_features(
    features
):

    """
    Combine layer2 and layer3 feature maps.

    layer2:
        [B, C1, H, W]

    layer3:
        [B, C2, H2, W2]

    layer3 is resized to layer2 spatial resolution.
    """

    if "layer2" not in features:

        raise KeyError(
            "layer2 is missing "
            "from feature extractor output."
        )

    if "layer3" not in features:

        raise KeyError(
            "layer3 is missing "
            "from feature extractor output."
        )

    layer2 = features[
        "layer2"
    ]

    layer3 = features[
        "layer3"
    ]

    # --------------------------------------------------------
    # Resize layer3
    # --------------------------------------------------------

    layer3 = F.interpolate(

        layer3,

        size=layer2.shape[-2:],

        mode="bilinear",

        align_corners=False,
    )

    # --------------------------------------------------------
    # Concatenate
    # --------------------------------------------------------

    feature_map = torch.cat(

        [
            layer2,
            layer3
        ],

        dim=1,
    )

    # --------------------------------------------------------
    # Feature map size
    # --------------------------------------------------------

    B, C, H, W = (
        feature_map.shape
    )

    print(
        f"Feature map: "
        f"B={B}, C={C}, H={H}, W={W}"
    )

    # --------------------------------------------------------
    # Convert to patches
    #
    # [B,C,H,W]
    #       ↓
    # [B,H,W,C]
    #       ↓
    # [B,H*W,C]
    # --------------------------------------------------------

    patches = (

        feature_map

        .permute(
            0,
            2,
            3,
            1
        )

        .reshape(
            B,
            H * W,
            C
        )
    )

    return (
        patches,
        feature_map,
    )


# ============================================================
# Resize anomaly map
# ============================================================

def resize_anomaly_map(

    anomaly_map,

    target_shape
):

    """
    Resize anomaly map to the exact
    Mel-spectrogram pixel dimensions.

    target_shape:
        (mel_bins, time_frames)
    """

    target_h = target_shape[0]
    target_w = target_shape[1]

    source_h = anomaly_map.shape[0]
    source_w = anomaly_map.shape[1]

    zoom_h = (
        target_h / source_h
    )

    zoom_w = (
        target_w / source_w
    )

    resized = zoom(

        anomaly_map,

        (
            zoom_h,
            zoom_w
        ),

        order=1,
    )

    # --------------------------------------------------------
    # Make absolutely sure the shape matches
    # --------------------------------------------------------

    resized = resized[
        :target_h,
        :target_w
    ]

    # Padding if needed because of rounding
    # --------------------------------------------------------

    if resized.shape != (
        target_h,
        target_w
    ):

        result = np.zeros(
            (
                target_h,
                target_w
            ),
            dtype=np.float32
        )

        h = min(
            target_h,
            resized.shape[0]
        )

        w = min(
            target_w,
            resized.shape[1]
        )

        result[
            :h,
            :w
        ] = resized[
            :h,
            :w
        ]

        resized = result

    return resized


# ============================================================
# Normalize anomaly map
# ============================================================

def normalize_anomaly_map(
    anomaly_map
):

    min_value = (
        anomaly_map.min()
    )

    max_value = (
        anomaly_map.max()
    )

    print(
        "\nAnomaly map range:"
    )

    print(
        f"min = {min_value:.6f}"
    )

    print(
        f"max = {max_value:.6f}"
    )

    if max_value > min_value:

        anomaly_map = (

            anomaly_map
            - min_value
        ) / (

            max_value
            - min_value
        )

    else:

        anomaly_map = np.zeros_like(
            anomaly_map
        )

    return anomaly_map


# ============================================================
# Visualize one sample
# ============================================================

def visualize(
    wav_path
):

    wav_path = Path(
        wav_path
    )

    print(
        "\n"
        "=================================================="
    )

    print(
        "PatchCore Anomaly Visualization"
    )

    print(
        "=================================================="
    )

    print(
        f"WAV: {wav_path}"
    )


    # ========================================================
    # Device
    # ========================================================

    device = get_device()

    print(
        f"Device: {device}"
    )


    # ========================================================
    # Check file
    # ========================================================

    if not wav_path.exists():

        raise FileNotFoundError(
            f"WAV file not found:\n"
            f"{wav_path}"
        )


    # ========================================================
    # Load Memory Bank
    # ========================================================

    memory_path = (

        MEMORY_ROOT
        / "MIMII_memory.npy"
    )

    if not memory_path.exists():

        raise FileNotFoundError(
            f"Memory bank not found:\n"
            f"{memory_path}"
        )


    memory = np.load(
        memory_path
    )

    print(
        f"Memory bank shape: "
        f"{memory.shape}"
    )


    # ========================================================
    # Create PatchCore
    # ========================================================

    patchcore = PatchCoreMemory(
        k=1
    )

    patchcore.fit(
        memory
    )


    # ========================================================
    # Feature Extractor
    # ========================================================

    extractor = (
        ResNetFeatureExtractor(
            device
        )
    )


    # ========================================================
    # Load WAV
    # ========================================================

    y, sr = load_audio(
        wav_path
    )

    duration = (
        len(y) / sr
    )

    print(
        f"Sample rate: {sr}"
    )

    print(
        f"Duration: {duration:.3f} sec"
    )


    # ========================================================
    # Mel Spectrogram
    # ========================================================

    mel_db = (
        create_mel_spectrogram(
            y,
            sr
        )
    )

    print(
        f"Mel shape: "
        f"{mel_db.shape}"
    )


    # ========================================================
    # Convert WAV → Tensor
    # ========================================================

    x = wav_to_tensor(

        wav_path,

        sr=SAMPLE_RATE,

        duration=DURATION,

        n_fft=N_FFT,

        hop_length=HOP_LENGTH,

        n_mels=N_MELS,

        image_size=IMAGE_SIZE,
    )


    # ========================================================
    # Extract CNN Features
    # ========================================================

    with torch.no_grad():

        features = extractor.extract(
            x
        )


    # ========================================================
    # Create Patch Embeddings
    # ========================================================

    patches, feature_map = (
        aggregate_features(
            features
        )
    )


    # ========================================================
    # Patch shape
    # ========================================================

    B, N, D = (
        patches.shape
    )

    print(
        f"Patch tensor: "
        f"{patches.shape}"
    )


    # ========================================================
    # Patch → NumPy
    # ========================================================

    patches_np = (

        patches

        .squeeze(0)

        .detach()

        .cpu()

        .numpy()

        .astype(
            np.float32
        )
    )


    # ========================================================
    # PatchCore prediction
    # ========================================================

    score, patch_scores = (

        patchcore.predict(
            patches_np
        )
    )


    # ========================================================
    # Patch grid
    # ========================================================

    _, _, H, W = (
        feature_map.shape
    )

    print(
        f"\nPatch grid: "
        f"{H} × {W}"
    )


    expected_patches = (
        H * W
    )

    if len(patch_scores) != (
        expected_patches
    ):

        raise RuntimeError(

            "Number of patch scores "
            "does not match feature map size.\n"

            f"patch_scores = "
            f"{len(patch_scores)}\n"

            f"H × W = "
            f"{H} × {W} = "
            f"{expected_patches}"
        )


    # ========================================================
    # Create anomaly map
    # ========================================================

    anomaly_map = (

        patch_scores

        .reshape(
            H,
            W
        )
    )


    # ========================================================
    # Print statistics
    # ========================================================

    print(
        "\n"
        "=================================================="
    )

    print(
        "PatchCore Result"
    )

    print(
        "=================================================="
    )

    print(
        f"Anomaly score : "
        f"{score:.6f}"
    )

    print(
        f"Patch min     : "
        f"{patch_scores.min():.6f}"
    )

    print(
        f"Patch max     : "
        f"{patch_scores.max():.6f}"
    )

    print(
        f"Patch mean    : "
        f"{patch_scores.mean():.6f}"
    )

    print(
        f"Patch std     : "
        f"{patch_scores.std():.6f}"
    )


    # ========================================================
    # Find most anomalous patch
    # ========================================================

    max_index = np.argmax(
        anomaly_map
    )

    max_y, max_x = np.unravel_index(

        max_index,

        anomaly_map.shape
    )

    max_patch_score = (

        anomaly_map[
            max_y,
            max_x
        ]
    )

    print(
        "\n"
        "Most anomalous patch"
    )

    print(
        f"Patch Y       : {max_y}"
    )

    print(
        f"Patch X       : {max_x}"
    )

    print(
        f"Patch score   : "
        f"{max_patch_score:.6f}"
    )


    # ========================================================
    # Normalize
    # ========================================================

    anomaly_map_norm = (
        normalize_anomaly_map(
            anomaly_map
        )
    )


    # ========================================================
    # Resize anomaly map
    # ========================================================

    anomaly_map_resized = (

        resize_anomaly_map(

            anomaly_map_norm,

            mel_db.shape
        )
    )


    print(
        "\nResized anomaly map:"
    )

    print(
        anomaly_map_resized.shape
    )


    print(
        "Mel spectrogram:"
    )

    print(
        mel_db.shape
    )


    # ========================================================
    # Verify same shape
    # ========================================================

    if anomaly_map_resized.shape != (
        mel_db.shape
    ):

        raise RuntimeError(

            "Mel spectrogram and "
            "anomaly map shapes do not match.\n"

            f"Mel: "
            f"{mel_db.shape}\n"

            f"Anomaly: "
            f"{anomaly_map_resized.shape}"
        )


    # ========================================================
    # ========================================================
    # Visualization
    # ========================================================
    # ========================================================


    fig, axes = plt.subplots(

        4,

        1,

        figsize=(14, 16)
    )


    # ========================================================
    # 1. Waveform
    # ========================================================

    librosa.display.waveshow(

        y,

        sr=sr,

        ax=axes[0]
    )

    axes[0].set_title(
        "1. Waveform"
    )

    axes[0].set_xlabel(
        "Time [s]"
    )


    # ========================================================
    # 2. Mel Spectrogram
    # ========================================================

    img1 = axes[1].imshow(

        mel_db,

        aspect="auto",

        origin="lower",

        interpolation="nearest",

        cmap="gray"
    )

    axes[1].set_title(
        "2. Log-Mel Spectrogram"
    )

    axes[1].set_xlabel(
        "Time Frame"
    )

    axes[1].set_ylabel(
        "Mel Bin"
    )

    fig.colorbar(

        img1,

        ax=axes[1],

        label="dB"
    )


    # ========================================================
    # 3. PatchCore Anomaly Map
    # ========================================================

    img2 = axes[2].imshow(

        anomaly_map_resized,

        aspect="auto",

        origin="lower",

        interpolation="nearest",

        cmap="jet",

        vmin=0.0,

        vmax=1.0
    )

    axes[2].set_title(

        "3. PatchCore Anomaly Map"
    )

    axes[2].set_xlabel(
        "Time Frame"
    )

    axes[2].set_ylabel(
        "Mel Bin"
    )

    fig.colorbar(

        img2,

        ax=axes[2],

        label="Normalized anomaly score"
    )


    # ========================================================
    # 4. Overlay
    # ========================================================

    # --------------------------------------------------------
    # Background:
    # Mel Spectrogram
    # --------------------------------------------------------

    axes[3].imshow(

        mel_db,

        aspect="auto",

        origin="lower",

        interpolation="nearest",

        cmap="gray",

        alpha=1.0
    )


    # --------------------------------------------------------
    # Foreground:
    # Anomaly map
    # --------------------------------------------------------

    img3 = axes[3].imshow(

        anomaly_map_resized,

        aspect="auto",

        origin="lower",

        interpolation="bilinear",

        cmap="jet",

        vmin=0.0,

        vmax=1.0,

        alpha=0.55
    )


    axes[3].set_title(

        "4. PatchCore Anomaly Overlay\n"

        f"Anomaly Score = "
        f"{score:.6f}"
    )

    axes[3].set_xlabel(
        "Time Frame"
    )

    axes[3].set_ylabel(
        "Mel Bin"
    )


    # --------------------------------------------------------
    # Colorbar
    # --------------------------------------------------------

    fig.colorbar(

        img3,

        ax=axes[3],

        label="Normalized anomaly score"
    )


    # ========================================================
    # Mark maximum anomaly location
    # ========================================================

    max_y_resized = (

        max_y
        / max(H - 1, 1)
        * (mel_db.shape[0] - 1)
    )

    max_x_resized = (

        max_x
        / max(W - 1, 1)
        * (mel_db.shape[1] - 1)
    )


    axes[3].scatter(

        max_x_resized,

        max_y_resized,

        s=150,

        facecolors="none",

        edgecolors="white",

        linewidths=2.0,

        marker="o"
    )


    axes[3].text(

        max_x_resized,

        max_y_resized,

        "  MAX",

        color="white",

        fontsize=12,

        fontweight="bold",

        verticalalignment="center"
    )


    # ========================================================
    # Layout
    # ========================================================

    plt.tight_layout()


    # ========================================================
    # Show
    # ========================================================

    plt.show()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    # ========================================================
    # Load test split
    # ========================================================

    normal_test, abnormal_test = (
        load_test_data()
    )


    print(
        f"Normal test samples   : "
        f"{len(normal_test)}"
    )

    print(
        f"Abnormal test samples : "
        f"{len(abnormal_test)}"
    )


    # ========================================================
    # Select one abnormal sample
    # ========================================================

    if len(abnormal_test) == 0:

        raise RuntimeError(
            "No abnormal test samples found."
        )


    # --------------------------------------------------------
    # First abnormal sample
    # --------------------------------------------------------

    test_file = Path(
        abnormal_test[10]   # 테스트용 샘플 파일의 선택 : 숫자 지정 (260814)
    )


    print(
        "\nSelected abnormal sample:"
    )

    print(
        test_file
    )


    # ========================================================
    # Visualize
    # ========================================================

    visualize(
        test_file
    )