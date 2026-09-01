"""
train_asd_pytorch.py로 학습한 모델을 불러와 테스트하는 스크립트.

두 가지 모드를 지원한다.

1) 재평가 모드 (기본): 학습 때와 동일한 데이터 분할을 재현하여
   정상 hold-out + abnormal 전체에 대한 AUC/pAUC를 다시 계산한다.
       python test_asd_pytorch.py

2) 새 파일 채점 모드: 라벨이 없는 새 wav 폴더를 대상으로 이상 점수와
   (학습 정상 데이터 기준 임계값 대비) 정상/비정상 판정을 계산해 CSV로 저장한다.
       python test_asd_pytorch.py --wav_dir ./data/MIMII/new_recordings
"""
import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from mimii_dataset import load_mimii, load_wav_fixed_length, _list_wavs
from asd_model import TwoBranchEmbeddingNet


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="asd_config.json",
                    help="train_asd_pytorch.py가 저장한 설정 파일")
    p.add_argument("--model_path", type=str, default="asd_model_pytorch.pt")
    p.add_argument("--centers_path", type=str, default="kmeans_centers.npy")
    p.add_argument("--data_root", type=str, default=None,
                    help="지정하지 않으면 config.json에 저장된 경로를 사용")
    p.add_argument("--wav_dir", type=str, default=None,
                    help="지정하면 이 폴더의 wav 파일들을 라벨 없이 채점 (재평가 모드 대신 실행)")
    p.add_argument("--threshold_percentile", type=float, default=90.0,
                    help="학습 정상 데이터 이상 점수 분포에서 정상/비정상 판정 임계값(백분위)")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--output_csv", type=str, default="test_results.csv")
    p.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def length_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(norm, 1e-12, None)


def load_model_and_centers(args, num_samples):
    device = torch.device(args.device)
    model = TwoBranchEmbeddingNet(num_samples=num_samples).to(device)
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    centers = np.load(args.centers_path)
    return model, centers, device


def extract_embeddings(model, device, wav: torch.Tensor, batch_size: int = 64):
    embs = []
    with torch.no_grad():
        for i in range(0, wav.shape[0], batch_size):
            batch = wav[i:i + batch_size].to(device)
            emb_fft, emb_mel = model.embed_clean(batch)
            emb = torch.cat([emb_fft, emb_mel], dim=-1)
            embs.append(emb.cpu().numpy())
    if not embs:
        return np.zeros((0, 256))
    return np.concatenate(embs, axis=0)


def compute_scores(model, device, centers, wav: torch.Tensor, batch_size: int = 64):
    embs = length_normalize(extract_embeddings(model, device, wav, batch_size))
    centers_n = length_normalize(centers)
    cos_dist = 1 - embs @ centers_n.T
    return cos_dist.min(axis=-1)


def compute_train_threshold(model, device, centers, data_root, sample_rate,
                             num_samples, percentile, batch_size):
    """학습에 쓰인 정상 데이터 전체(hold-out 제외 없이)로 임계값을 계산."""
    normal_files = _list_wavs(os.path.join(data_root, "normal"))
    if len(normal_files) == 0:
        return None
    wavs = np.stack([
        load_wav_fixed_length(f, sample_rate, num_samples, random_crop=False)
        for f in normal_files
    ], axis=0)
    wav_t = torch.from_numpy(wavs).float()
    scores = compute_scores(model, device, centers, wav_t, batch_size)
    return float(np.percentile(scores, percentile))


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)
    sample_rate = config["sample_rate"]
    num_samples = config["num_samples"]
    data_root = args.data_root or config["data_root"]
    seed = config.get("seed", 0)

    model, centers, device = load_model_and_centers(args, num_samples)

    if args.wav_dir is not None:
        # ------------------------------------------------------------
        # 모드 2: 라벨 없는 새 wav 폴더 채점
        # ------------------------------------------------------------
        files = _list_wavs(args.wav_dir)
        if len(files) == 0:
            print(f"{args.wav_dir} 에서 wav 파일을 찾지 못했습니다.")
            return
        wavs = np.stack([
            load_wav_fixed_length(f, sample_rate, num_samples, random_crop=False)
            for f in files
        ], axis=0)
        wav_t = torch.from_numpy(wavs).float()
        scores = compute_scores(model, device, centers, wav_t, args.batch_size)

        threshold = compute_train_threshold(
            model, device, centers, data_root, sample_rate, num_samples,
            args.threshold_percentile, args.batch_size)

        with open(args.output_csv, "w") as f:
            f.write("file,anomaly_score,decision\n")
            for path, score in zip(files, scores):
                if threshold is not None:
                    decision = "abnormal" if score > threshold else "normal"
                else:
                    decision = "n/a"
                f.write(f"{path},{score:.6f},{decision}\n")
                print(f"{os.path.basename(path)}: score={score:.4f}  -> {decision}")

        if threshold is not None:
            print(f"\n임계값(정상 데이터 {args.threshold_percentile}백분위) = {threshold:.4f}")
        print(f"결과 저장: {args.output_csv}")
        return

    # ------------------------------------------------------------------
    # 모드 1: 학습 때와 동일한 분할을 재현하여 AUC/pAUC 재계산
    # ------------------------------------------------------------------
    data = load_mimii(data_root, target_sr=sample_rate,
                       duration_sec=config["duration_sec"], seed=seed)
    test_wav = torch.from_numpy(data["test_wav"]).float()
    test_label = data["test_label"]
    test_files = data["test_files"]

    scores = compute_scores(model, device, centers, test_wav, args.batch_size)

    with open(args.output_csv, "w") as f:
        f.write("file,label,anomaly_score\n")
        for path, label, score in zip(test_files, test_label, scores):
            tag = "abnormal" if label == 1 else "normal"
            f.write(f"{path},{tag},{score:.6f}\n")

    if len(np.unique(test_label)) < 2:
        print("비정상 샘플이 없어 AUC를 계산할 수 없습니다.")
    else:
        auc = roc_auc_score(test_label, scores)
        p_auc = roc_auc_score(test_label, scores, max_fpr=0.1)
        print(f"AUC  = {auc * 100:.2f}%")
        print(f"pAUC = {p_auc * 100:.2f}%")

    print(f"결과 저장: {args.output_csv}")


if __name__ == "__main__":
    main()
