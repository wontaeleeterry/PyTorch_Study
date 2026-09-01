"""
논문의 두 갈래(spectrum / spectrogram+TMN) 임베딩 네트워크를 PyTorch로 재구현.
원본은 5단 Squeeze-and-Excitation ResNet(2D)이지만, 여기서는 표현력을 유지하면서
가독성을 위해 3단으로 축소했다. 필요하면 _ResBlock2D를 더 쌓아 원본 깊이로 늘릴 수 있다.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcite1D(nn.Module):
    def __init__(self, channels: int, ratio: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(channels, max(channels // ratio, 1), bias=False)
        self.fc2 = nn.Linear(max(channels // ratio, 1), channels, bias=False)

    def forward(self, x):  # x: (B, C, L)
        s = x.mean(dim=-1)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s.unsqueeze(-1)


class SqueezeExcite2D(nn.Module):
    def __init__(self, channels: int, ratio: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(channels, max(channels // ratio, 1), bias=False)
        self.fc2 = nn.Linear(max(channels // ratio, 1), channels, bias=False)

    def forward(self, x):  # x: (B, C, H, W)
        s = x.mean(dim=(-1, -2))
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s.unsqueeze(-1).unsqueeze(-1)


class SpectrumBranch(nn.Module):
    """전체 magnitude spectrum(FFT)을 입력받는 1D CNN 서브네트워크 -> 128차원 임베딩."""

    def __init__(self, num_bins: int, emb_dim: int = 128):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 128, kernel_size=256, stride=64, padding=128, bias=False)
        self.se1 = SqueezeExcite1D(128)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=64, stride=32, padding=32, bias=False)
        self.se2 = SqueezeExcite1D(128)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=16, stride=4, padding=8, bias=False)
        self.se3 = SqueezeExcite1D(128)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, num_bins)
            flat_dim = self._conv_forward(dummy).flatten(1).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(flat_dim, 128, bias=False), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 128, bias=False), nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.out = nn.Linear(128, emb_dim, bias=False)

    def _conv_forward(self, x):
        x = self.se1(F.relu(self.conv1(x)))
        x = self.se2(F.relu(self.conv2(x)))
        x = self.se3(F.relu(self.conv3(x)))
        return x

    def forward(self, spectrum: torch.Tensor):  # spectrum: (B, num_bins)
        x = spectrum.unsqueeze(1)
        x = self._conv_forward(x)
        x = x.flatten(1)
        x = self.fc(x)
        return self.out(x)


class _ResBlock2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.se = SqueezeExcite2D(out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.MaxPool2d(2, ceil_mode=True) if stride == 2 else nn.Identity(),
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
            )

    def forward(self, x):
        identity = self.downsample(x) if self.downsample is not None else x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        out = self.se(out)
        out = out + identity
        out = self.bn2(out)
        return out


class SpectrogramBranch(nn.Module):
    """STFT magnitude spectrogram(+TMN)을 입력받는 2D CNN 서브네트워크 -> 128차원 임베딩."""

    def __init__(self, emb_dim: int = 128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.block1 = _ResBlock2D(16, 32, stride=2)
        self.block2 = _ResBlock2D(32, 64, stride=2)
        self.block3 = _ResBlock2D(64, 128, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn_flat = nn.BatchNorm1d(128)
        self.out = nn.Linear(128, emb_dim, bias=False)

    def forward(self, spec_tmn: torch.Tensor):  # (B, 1, F, T), 이미 TMN 적용된 상태
        x = self.stem(spec_tmn)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x).flatten(1)
        x = self.bn_flat(x)
        return self.out(x)


def compute_spectrum(wav: torch.Tensor) -> torch.Tensor:
    """(B, L) -> (B, L//2) magnitude spectrum (전체 파형의 단일 FFT)."""
    spec = torch.fft.rfft(wav, dim=-1).abs()
    return spec[:, : wav.shape[-1] // 2]


def compute_spectrogram(wav: torch.Tensor, n_fft: int = 1024, hop: int = 512,
                         f_min_bin: int = 13, f_max_bin: int = None) -> torch.Tensor:
    """(B, L) -> (B, 1, F, T) magnitude spectrogram (선택적 주파수 대역 crop)."""
    window = torch.hann_window(n_fft, device=wav.device)
    spec = torch.stft(wav, n_fft=n_fft, hop_length=hop, window=window,
                       return_complex=True, center=False)
    spec = spec.abs()
    if f_max_bin is None:
        f_max_bin = spec.shape[1]
    spec = spec[:, f_min_bin:f_max_bin, :]
    return spec.unsqueeze(1)


class TwoBranchEmbeddingNet(nn.Module):
    def __init__(self, num_samples: int, emb_dim: int = 128,
                 n_fft: int = 1024, hop: int = 512):
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop
        num_spectrum_bins = num_samples // 2
        self.spectrum_branch = SpectrumBranch(num_spectrum_bins, emb_dim)
        self.spectrogram_branch = SpectrogramBranch(emb_dim)

    def embed_clean(self, wav: torch.Tensor):
        """SSL 증강 없이 순수 임베딩만 추출 (추론/백엔드 용)."""
        emb_fft = self.spectrum_branch(compute_spectrum(wav))
        spec = compute_spectrogram(wav, self.n_fft, self.hop)
        spec = spec - spec.mean(dim=-1, keepdim=True)  # temporal mean normalization
        emb_mel = self.spectrogram_branch(spec)
        return emb_fft, emb_mel

    def spectrum_embedding(self, wav: torch.Tensor):
        return self.spectrum_branch(compute_spectrum(wav))

    def spectrogram_embedding_from_spec(self, spec: torch.Tensor):
        spec = spec - spec.mean(dim=-1, keepdim=True)
        return self.spectrogram_branch(spec)

    def raw_spectrogram(self, wav: torch.Tensor):
        return compute_spectrogram(wav, self.n_fft, self.hop)
