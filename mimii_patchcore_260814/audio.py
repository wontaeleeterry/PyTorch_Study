import librosa
import numpy as np
import torch
import torch.nn.functional as F


def load_audio(
    path,
    sr=16000,
    duration=10.0,
):
    """
    Load audio and force it to a fixed duration.
    """

    y, _ = librosa.load(
        path,
        sr=sr,
        mono=True,
    )

    target_length = int(sr * duration)

    if len(y) < target_length:

        y = np.pad(
            y,
            (0, target_length - len(y)),
        )

    else:

        y = y[:target_length]

    return y.astype(np.float32)


def make_logmel(
    y,
    sr=16000,
    n_fft=1024,
    hop_length=512,
    n_mels=128,
):
    """
    Waveform -> Log-Mel Spectrogram
    """

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )

    logmel = librosa.power_to_db(
        mel,
        ref=np.max,
    )

    return logmel.astype(np.float32)


def spectrogram_to_tensor(
    spectrogram,
    image_size=224,
):
    """
    [Mel, Time]
        ->
    [1, 3, 224, 224]
    """

    spec = spectrogram.copy()

    # Normalize to [0, 1]
    spec_min = spec.min()
    spec_max = spec.max()

    spec = (
        spec - spec_min
    ) / (
        spec_max - spec_min + 1e-8
    )

    x = torch.from_numpy(spec)

    # [H, W]
    x = x.unsqueeze(0).unsqueeze(0)

    # [1, 1, H, W]
    x = F.interpolate(
        x,
        size=(
            image_size,
            image_size,
        ),
        mode="bilinear",
        align_corners=False,
    )

    # [1, 1, H, W]
    x = x.repeat(
        1,
        3,
        1,
        1,
    )

    # ImageNet pretrained ResNet normalization
    mean = torch.tensor(
        [0.485, 0.456, 0.406]
    ).view(1, 3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225]
    ).view(1, 3, 1, 1)

    x = (x - mean) / std

    return x


def wav_to_tensor(
    path,
    sr=16000,
    duration=10.0,
    n_fft=1024,
    hop_length=512,
    n_mels=128,
    image_size=224,
):
    """
    Complete preprocessing pipeline.
    """

    y = load_audio(
        path,
        sr=sr,
        duration=duration,
    )

    spec = make_logmel(
        y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )

    x = spectrogram_to_tensor(
        spec,
        image_size=image_size,
    )

    return x