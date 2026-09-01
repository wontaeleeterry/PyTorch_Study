"""
./data/MIMII/normal/*.wav  와  ./data/MIMII/abnormal/*.wav 구조에서
오디오를 읽어 고정 길이 파형 텐서로 만드는 유틸리티.

원본(wilkinghoff/ssl4asd)의 DCASE 폴더 구조(기계타입/ID/속성 메타정보,
train/test, source/target 도메인 구분)를 가정한 파서를 대체한다.
여기서는 메타정보가 없으므로 단일 클래스(label=0)로 취급한다.
"""
import os
import glob
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm


def _list_wavs(root: str):
    return sorted(glob.glob(os.path.join(root, "**", "*.wav"), recursive=True))


def load_wav_fixed_length(path: str, target_sr: int, num_samples: int, random_crop: bool = False):
    """다른 스크립트(test_asd_pytorch.py 등)에서도 재사용할 수 있게 공개 함수로 노출."""
    return _load_and_fix_length(path, target_sr, num_samples, random_crop)


def _load_and_fix_length(path: str, target_sr: int, num_samples: int, random_crop: bool):
    wav, sr = sf.read(path)
    wav = librosa.core.to_mono(wav.T).T if wav.ndim > 1 else wav
    if sr != target_sr:
        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=target_sr)
    wav = wav.astype(np.float32)

    if wav.shape[0] >= num_samples:
        if random_crop:
            offset = np.random.randint(0, wav.shape[0] - num_samples + 1)
        else:
            offset = 0
        wav = wav[offset:offset + num_samples]
    else:
        reps = int(np.ceil(num_samples / wav.shape[0]))
        wav = np.tile(wav, reps)[:num_samples]
    return wav


def load_mimii(data_root: str, target_sr: int = 16000, duration_sec: float = 10.0,
               test_ratio: float = 0.2, seed: int = 0):
    """
    data_root: './data/MIMII' 를 가리킴 (하위에 normal/, abnormal/ 존재)
    반환: dict(train_wav, train_files, test_wav, test_files, test_label)
          test_label: 0=normal, 1=abnormal
    """
    num_samples = int(target_sr * duration_sec)
    normal_files = _list_wavs(os.path.join(data_root, "normal"))
    abnormal_files = _list_wavs(os.path.join(data_root, "abnormal"))

    if len(normal_files) == 0:
        raise FileNotFoundError(f"No wav files found under {data_root}/normal")
    if len(abnormal_files) == 0:
        print(f"[warning] No wav files found under {data_root}/abnormal — "
              f"AUC 계산 시 비정상 샘플이 없어 평가가 불가능합니다.")

    rng = np.random.RandomState(seed)
    normal_files = np.array(normal_files)
    perm = rng.permutation(len(normal_files))
    normal_files = normal_files[perm]
    n_test = max(1, int(len(normal_files) * test_ratio))
    test_normal_files = normal_files[:n_test]
    train_normal_files = normal_files[n_test:]

    def _load_all(files, random_crop):
        out = []
        for f in tqdm(files, desc=f"loading {len(files)} wavs"):
            out.append(_load_and_fix_length(f, target_sr, num_samples, random_crop))
        return np.stack(out, axis=0) if len(out) > 0 else np.zeros((0, num_samples), dtype=np.float32)

    print("Loading training (normal-only) data ...")
    train_wav = _load_all(train_normal_files, random_crop=False)

    print("Loading test data (normal hold-out + abnormal) ...")
    test_normal_wav = _load_all(test_normal_files, random_crop=False)
    test_abnormal_wav = _load_all(abnormal_files, random_crop=False)

    test_wav = np.concatenate([test_normal_wav, test_abnormal_wav], axis=0)
    test_files = np.concatenate([test_normal_files, np.array(abnormal_files)], axis=0)
    test_label = np.concatenate([
        np.zeros(len(test_normal_files), dtype=np.int64),
        np.ones(len(abnormal_files), dtype=np.int64),
    ])

    return {
        "train_wav": train_wav,
        "train_files": train_normal_files,
        "test_wav": test_wav,
        "test_files": test_files,
        "test_label": test_label,
        "num_samples": num_samples,
    }
