"""
./data/MIMII/normal, ./data/MIMII/abnormal 구조의 데이터를 이용해
Wilkinghoff (2024) "Self-Supervised Learning for Anomalous Sound Detection"의
파이프라인(Mixup + StatEx + FeatEx + Sub-cluster AdaCos)을 PyTorch로 학습/평가한다.

실행 예:
    python train_asd_pytorch.py --data_root ./data/MIMII --epochs 10
"""
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score

from mimii_dataset import load_mimii
from asd_model import TwoBranchEmbeddingNet
from sub_cluster_adacos import SubClusterAdaCos
from ssl_augmentations import (
    mixup_waveform, statex_spectrogram, featex_embeddings, soft_cross_entropy,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="./data/MIMII",
                    help="하위에 normal/, abnormal/ 폴더가 있는 경로")
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--duration_sec", type=float, default=10.0)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--n_subclusters", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--mixup_prob", type=float, default=0.5)
    p.add_argument("--statex_prob", type=float, default=0.5)
    p.add_argument("--featex_prob", type=float, default=0.5)
    p.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def length_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(norm, 1e-12, None)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    data = load_mimii(args.data_root, target_sr=args.sample_rate,
                       duration_sec=args.duration_sec, seed=args.seed)
    train_wav = torch.from_numpy(data["train_wav"]).float()
    test_wav = torch.from_numpy(data["test_wav"]).float()
    test_label = data["test_label"]
    num_samples = data["num_samples"]

    # 메타정보(기계타입/ID/속성)가 없으므로 단일 클래스로 취급.
    num_classes = 1
    y_train = torch.zeros(train_wav.shape[0], num_classes)
    y_train[:, 0] = 1.0

    train_loader = DataLoader(
        TensorDataset(train_wav, y_train), batch_size=args.batch_size,
        shuffle=True, drop_last=True,
    )

    model = TwoBranchEmbeddingNet(num_samples=num_samples).to(device)

    # 손실 헤드 3개: (1) 원본 클래스(중심 고정), (2) StatEx+FeatEx 결합(9배 클래스, 중심 학습)
    emb_dim_concat = 128 * 2
    head_main = SubClusterAdaCos(emb_dim_concat, num_classes,
                                  args.n_subclusters, trainable=False).to(device)
    head_ssl = SubClusterAdaCos(emb_dim_concat, num_classes * 9,
                                 args.n_subclusters, trainable=True).to(device)

    params = list(model.parameters()) + list(head_ssl.parameters())
    if any(p.requires_grad for p in head_main.parameters()):
        params += list(head_main.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)

    print(f"학습 샘플 수: {train_wav.shape[0]}, 테스트 샘플 수: {test_wav.shape[0]} "
          f"(정상 {int((test_label == 0).sum())} / 비정상 {int((test_label == 1).sum())})")

    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        n_batches = 0
        for wav, y in train_loader:
            wav, y = wav.to(device), y.to(device)

            # 1) Mixup: 원파형 단계에서 적용
            wav_mix, y_mix = mixup_waveform(wav, y, prob=args.mixup_prob, training=True)

            # 2) 스펙트럼 브랜치 (Mixup 이후 파형 기준)
            emb_fft = model.spectrum_embedding(wav_mix)

            # 3) 스펙트로그램 브랜치: StatEx는 CNN 이전, TMN 이전 단계에 적용
            spec_raw = model.raw_spectrogram(wav_mix)
            spec_statex, y_statex = statex_spectrogram(
                spec_raw, y_mix, prob=args.statex_prob, training=True)
            emb_mel = model.spectrogram_embedding_from_spec(spec_statex)

            # 4) FeatEx: StatEx 이후의 두 임베딩을 교환하여 9배 클래스 라벨 생성
            emb_fft_ssl, emb_mel_ssl, y_featex = featex_embeddings(
                emb_fft, emb_mel, y_statex, prob=args.featex_prob, training=True)

            # 손실 (1): 원본 개념의 지도 손실 (Mixup 라벨, 중심 고정)
            x_main = torch.cat([emb_fft, emb_mel], dim=-1)
            loss_main = head_main(x_main, y_mix)

            # 손실 (2): StatEx+FeatEx 결합 SSL 손실 (9배 클래스, 중심 학습)
            x_ssl = torch.cat([emb_fft_ssl, emb_mel_ssl], dim=-1)
            loss_ssl = head_ssl(x_ssl, y_featex)

            loss = loss_main + loss_ssl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        print(f"[epoch {epoch + 1}/{args.epochs}] loss = {epoch_loss / max(n_batches, 1):.4f}")

    # ------------------------------------------------------------------
    # 임베딩 추출 (SSL 증강 없이 순수 임베딩) + k-means 배경 모델 + 코사인 거리
    # ------------------------------------------------------------------
    model.eval()

    def extract_embeddings(wav_tensor: torch.Tensor, batch_size: int = 64):
        embs = []
        with torch.no_grad():
            for i in range(0, wav_tensor.shape[0], batch_size):
                batch = wav_tensor[i:i + batch_size].to(device)
                emb_fft, emb_mel = model.embed_clean(batch)
                emb = torch.cat([emb_fft, emb_mel], dim=-1)
                embs.append(emb.cpu().numpy())
        return np.concatenate(embs, axis=0) if embs else np.zeros((0, emb_dim_concat))

    train_embs = length_normalize(extract_embeddings(train_wav))
    test_embs = length_normalize(extract_embeddings(test_wav))

    kmeans = KMeans(n_clusters=min(args.n_subclusters, train_embs.shape[0]),
                     random_state=args.seed, n_init=10).fit(train_embs)
    centers = length_normalize(kmeans.cluster_centers_)

    cos_dist = 1 - test_embs @ centers.T
    anomaly_score = cos_dist.min(axis=-1)

    if len(np.unique(test_label)) < 2:
        print("비정상 샘플이 없어 AUC를 계산할 수 없습니다. "
              "./data/MIMII/abnormal 폴더에 wav 파일을 넣어주세요.")
    else:
        auc = roc_auc_score(test_label, anomaly_score)
        p_auc = roc_auc_score(test_label, anomaly_score, max_fpr=0.1)
        print(f"AUC  = {auc * 100:.2f}%")
        print(f"pAUC = {p_auc * 100:.2f}%")

    np.save("test_anomaly_scores.npy", anomaly_score)
    np.save("test_labels.npy", test_label)
    np.save("kmeans_centers.npy", centers)
    torch.save(model.state_dict(), "asd_model_pytorch.pt")

    import json
    config = {
        "sample_rate": args.sample_rate,
        "duration_sec": args.duration_sec,
        "num_samples": num_samples,
        "n_subclusters": args.n_subclusters,
        "seed": args.seed,
        "data_root": args.data_root,
    }
    with open("asd_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("결과 저장: test_anomaly_scores.npy, test_labels.npy, kmeans_centers.npy, "
          "asd_model_pytorch.pt, asd_config.json")


if __name__ == "__main__":
    main()
