"""
논문 3장의 세 가지 SSL 기법을 PyTorch로 구현.

원본 Keras 구현(mixup_layer.py, statex_aug_layer_classwise.py,
openl3_idea_aug_layer_classwise.py)과 동일하게, 배치를 반전(flip)시켜
"다른 샘플"과 짝을 지은 뒤 학습 시에만(prob 확률로) 적용한다.
평가/추론 시에는 항등함수로 동작한다.
"""
import torch


def soft_cross_entropy(logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    """target_probs: 정규화된 확률 분포(합이 1일 필요는 없음, mixup/statex/featex의
    라벨은 부분합이 1보다 작을 수 있음 - 그대로 가중합으로 사용)."""
    log_probs = torch.log_softmax(logits, dim=-1)
    return -(target_probs * log_probs).sum(dim=-1).mean()


def mixup_waveform(wav: torch.Tensor, y: torch.Tensor, prob: float = 0.5,
                    training: bool = True):
    """
    wav: (B, L), y: (B, C) one-hot/soft label
    적용되면 두 샘플을 선형보간. 라벨도 동일 비율로 보간(라벨 차원은 변하지 않음).
    """
    if not training:
        return wav, y

    batch_size = wav.shape[0]
    perm = torch.randperm(batch_size, device=wav.device)
    lam = torch.rand(batch_size, device=wav.device)
    apply_mask = (torch.rand(batch_size, device=wav.device) < prob).float()
    lam = lam * apply_mask + 1.0 * (1 - apply_mask)  # 적용 안 하면 lam=1 (원본 그대로)

    lam_wav = lam.view(-1, 1)
    lam_y = lam.view(-1, 1)

    wav_new = lam_wav * wav + (1 - lam_wav) * wav[perm]
    y_new = lam_y * y + (1 - lam_y) * y[perm]
    return wav_new, y_new


def statex_spectrogram(spec: torch.Tensor, y: torch.Tensor, prob: float = 0.5,
                        training: bool = True, eps: float = 1e-8):
    """
    spec: (B, 1, F, T) 크기의 magnitude spectrogram (TMN 이전 단계)
    시간축(T)을 따라 1차/2차 통계량을 다른 샘플과 교환하여 가짜 이상 클래스 생성.
    y: (B, C) -> 반환 라벨은 (B, 3C): [0, 0.5*y1, 0.5*y2] (미적용 시 [y1, 0, 0])
    """
    num_classes = y.shape[-1]
    zeros = torch.zeros_like(y)

    if not training:
        return spec, torch.cat([y, zeros, zeros], dim=-1)

    batch_size = spec.shape[0]
    perm = torch.randperm(batch_size, device=spec.device)
    apply_mask = (torch.rand(batch_size, device=spec.device) < prob)

    # 시간축(T, 마지막 차원)에 대한 평균/표준편차
    mu1 = spec.mean(dim=-1, keepdim=True)
    sigma1 = spec.std(dim=-1, keepdim=True) + eps
    mu2 = mu1[perm]
    sigma2 = sigma1[perm]

    spec_statex = (spec - mu1) / sigma1 * sigma2 + mu2

    mask = apply_mask.view(-1, 1, 1, 1).float()
    spec_new = mask * spec_statex + (1 - mask) * spec

    y1, y2 = y, y[perm]
    mask_y = apply_mask.view(-1, 1).float()
    y_new = torch.cat([
        (1 - mask_y) * y1,
        mask_y * 0.5 * y1,
        mask_y * 0.5 * y2,
    ], dim=-1)
    return spec_new, y_new


def featex_embeddings(emb_a: torch.Tensor, emb_b: torch.Tensor, y: torch.Tensor,
                       prob: float = 0.5, training: bool = True):
    """
    Feature exchange (FeatEx, 제안 기법).
    emb_a, emb_b: (B, D) 두 서브네트워크의 임베딩.
    한 샘플의 emb_a와 다른 샘플의 emb_b를 교환해 새 임베딩쌍 생성.
    y: (B, C) -> 반환 라벨은 (B, 3C), C는 입력받은 y의 차원(이미 StatEx로 3배 된 경우 그대로 유지되어
    최종적으로 9배가 됨).
    """
    num_classes = y.shape[-1]
    zeros = torch.zeros_like(y)

    if not training:
        return emb_a, emb_b, torch.cat([y, zeros, zeros], dim=-1)

    batch_size = emb_a.shape[0]
    perm = torch.randperm(batch_size, device=emb_a.device)
    apply_mask = (torch.rand(batch_size, device=emb_a.device) < prob)

    emb_a_new = emb_a
    emb_b_new = torch.where(apply_mask.view(-1, 1), emb_b[perm], emb_b)

    y1, y2 = y, y[perm]
    mask_y = apply_mask.view(-1, 1).float()
    y_new = torch.cat([
        (1 - mask_y) * y1,
        mask_y * 0.5 * y1,
        mask_y * 0.5 * y2,
    ], dim=-1)
    return emb_a_new, emb_b_new, y_new
