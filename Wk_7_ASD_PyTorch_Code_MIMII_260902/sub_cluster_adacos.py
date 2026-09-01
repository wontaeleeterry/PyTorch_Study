"""
Sub-cluster AdaCos (Wilkinghoff, IJCNN 2021)의 PyTorch 재구현.

- 클래스마다 n_subclusters개의 중심 벡터를 두고, 임베딩과 모든 중심 벡터의
  코사인 유사도 중 "클래스 내 최댓값"을 그 클래스에 대한 유사도로 사용.
- AdaCos 방식으로 스케일 s를 학습 없이 배치 통계로 적응적으로 갱신.
- trainable=False 이면 중심 벡터는 랜덤 초기화 후 고정(gradient 차단).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SubClusterAdaCos(nn.Module):
    def __init__(self, embedding_dim: int, n_classes: int, n_subclusters: int = 16,
                 trainable: bool = True):
        super().__init__()
        self.n_classes = n_classes
        self.n_subclusters = n_subclusters
        self.trainable = trainable

        centers = torch.randn(n_classes * n_subclusters, embedding_dim)
        centers = F.normalize(centers, dim=-1)
        if trainable:
            self.centers = nn.Parameter(centers)
        else:
            self.register_buffer("centers", centers)

        init_s = math.sqrt(2) * math.log(max(n_classes - 1, 1))
        self.register_buffer("s", torch.tensor(float(init_s)))

    def forward(self, embeddings: torch.Tensor, target_probs: torch.Tensor):
        """
        embeddings: (B, D)
        target_probs: (B, n_classes) soft label (mixup/statex/featex 조합 라벨 포함)
        반환: scalar loss
        """
        x = F.normalize(embeddings, dim=-1)
        w = F.normalize(self.centers, dim=-1) if self.trainable else self.centers

        cos_all = x @ w.t()  # (B, n_classes*n_subclusters)
        cos_all = cos_all.view(-1, self.n_classes, self.n_subclusters)
        cos_class, _ = cos_all.max(dim=-1)  # (B, n_classes)

        hard_target = target_probs.argmax(dim=-1)

        with torch.no_grad():
            theta = torch.acos(cos_class.clamp(-1 + 1e-7, 1 - 1e-7))
            b_avg = torch.exp(self.s * cos_class)
            b_avg = torch.where(
                F.one_hot(hard_target, self.n_classes).bool(),
                torch.zeros_like(b_avg),
                b_avg,
            ).sum(dim=-1).mean()

            theta_pos = theta.gather(1, hard_target.view(-1, 1)).squeeze(1)
            correct = cos_class.argmax(dim=-1) == hard_target
            if correct.any():
                theta_med = theta_pos[correct].median()
            else:
                theta_med = theta_pos.median()
            theta_med = torch.clamp(theta_med, max=math.pi / 4)

            new_s = torch.log(b_avg.clamp(min=1e-12)) / torch.cos(theta_med).clamp(min=1e-4)
            if torch.isfinite(new_s) and new_s > 0:
                self.s = new_s.detach()

        logits = self.s * cos_class
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(target_probs * log_probs).sum(dim=-1).mean()
        return loss
