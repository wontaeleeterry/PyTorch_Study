import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm

import faiss


# ============================================================
# Feature map -> patch embedding
# ============================================================

def feature_map_to_patches(
    feature,
):
    """
    [B, C, H, W]
        ->
    [B, H*W, C]
    """

    b, c, h, w = feature.shape

    feature = feature.permute(
        0,
        2,
        3,
        1,
    )

    feature = feature.reshape(
        b,
        h * w,
        c,
    )

    return feature


def aggregate_features(
    features,
):
    """
    Combine layer2 and layer3.

    layer2:
        [B, C2, H2, W2]

    layer3:
        [B, C3, H3, W3]

    Returns:
        [B, H2*W2, C2+C3]
    """

    layer2 = features["layer2"]

    layer3 = features["layer3"]

    # --------------------------------------------------------
    # layer2
    # --------------------------------------------------------

    layer2_patches = (
        feature_map_to_patches(
            layer2
        )
    )

    b, n, c2 = (
        layer2_patches.shape
    )

    h = layer2.shape[2]
    w = layer2.shape[3]

    # --------------------------------------------------------
    # layer3 -> spatially resize
    # --------------------------------------------------------

    layer3 = F.interpolate(
        layer3,
        size=(h, w),
        mode="bilinear",
        align_corners=False,
    )

    layer3_patches = (
        feature_map_to_patches(
            layer3
        )
    )

    # --------------------------------------------------------
    # concatenate
    # --------------------------------------------------------

    features = torch.cat(
        [
            layer2_patches,
            layer3_patches,
        ],
        dim=-1,
    )

    return features


# ============================================================
# Coreset
# ============================================================

class CoresetSampler:
    """
    PatchCore의 approximate greedy coreset selection.

    Greedy k-center (minimax facility location):
    현재까지 선택된 center 집합과의 거리(최근접 center까지의 거리)가
    가장 먼 점을 반복적으로 새 center로 추가한다.

    연산량을 줄이기 위해:
      1) Johnson-Lindenstrauss random projection으로 차원을 축소해서
         거리 계산 비용을 낮추고,
      2) 매 반복마다 전체 pairwise distance를 다시 계산하지 않고
         min_distance(각 점 -> 가장 가까운 center)만 갱신한다.
         (O(target * n) matrix 연산, target 회의 O(n) 갱신)

    method="random"으로 두면 기존의 단순 랜덤 샘플링도 그대로 사용할 수 있다
    (속도 비교/디버깅용).
    """

    def __init__(
        self,
        ratio=0.01,
        random_seed=42,
        method="greedy",
        projection_dim=128,
        device=None,
    ):

        self.ratio = ratio

        self.random_seed = (
            random_seed
        )

        self.method = method

        self.projection_dim = (
            projection_dim
        )

        self.device = (
            device
            if device is not None
            else torch.device("cpu")
        )

    # ------------------------------------------------------------
    # Random projection (Johnson-Lindenstrauss)
    # ------------------------------------------------------------

    def _random_projection(
        self,
        features,
    ):
        """
        [N, D] -> [N, projection_dim]

        projection_dim이 None이거나 원래 차원보다 크면
        projection을 적용하지 않는다.
        """

        dim = features.shape[1]

        if (
            self.projection_dim is None
            or self.projection_dim >= dim
        ):

            return features

        generator = (
            torch.Generator()
            .manual_seed(self.random_seed)
        )

        projection_matrix = (
            torch.randn(
                dim,
                self.projection_dim,
                generator=generator,
            )
            / (self.projection_dim ** 0.5)
        )

        projection_matrix = (
            projection_matrix
            .to(features.device)
        )

        return features @ projection_matrix

    # ------------------------------------------------------------
    # Greedy k-center coreset
    # ------------------------------------------------------------

    def _greedy_sample(
        self,
        features_np,
        target,
    ):

        n = len(features_np)

        features_t = (
            torch.from_numpy(features_np)
            .to(self.device)
        )

        proj = self._random_projection(
            features_t
        )

        rng = np.random.default_rng(
            self.random_seed
        )

        # ----------------------------------------------------
        # 첫 center는 무작위로 선택
        # ----------------------------------------------------

        first_idx = int(
            rng.integers(0, n)
        )

        selected_indices = [
            first_idx
        ]

        min_distances = (
            torch.cdist(
                proj,
                proj[first_idx : first_idx + 1],
            )
            .squeeze(1)
        )

        # 이미 선택된 점이 다시 뽑히지 않도록 마스킹
        min_distances[first_idx] = -1.0

        # ----------------------------------------------------
        # 반복적으로 min_distance가 최대인 점을 center로 추가
        # ----------------------------------------------------

        for _ in tqdm(
            range(1, target),

            desc="Greedy coreset sampling",
        ):

            next_idx = int(
                torch.argmax(
                    min_distances
                ).item()
            )

            selected_indices.append(
                next_idx
            )

            new_distances = (
                torch.cdist(
                    proj,
                    proj[next_idx : next_idx + 1],
                )
                .squeeze(1)
            )

            min_distances = (
                torch.minimum(
                    min_distances,
                    new_distances,
                )
            )

            min_distances[next_idx] = -1.0

        return np.array(
            selected_indices
        )

    # ------------------------------------------------------------
    # Random sampling (fallback / 비교용)
    # ------------------------------------------------------------

    def _random_sample(
        self,
        n,
        target,
    ):

        rng = np.random.default_rng(
            self.random_seed
        )

        return rng.choice(
            n,
            size=target,
            replace=False,
        )

    # ------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------

    def sample(
        self,
        features,
    ):

        features = np.asarray(
            features,
            dtype=np.float32,
        )

        n = len(features)

        target = max(
            1,
            int(n * self.ratio)
        )

        if target >= n:

            return features

        if self.method == "greedy":

            indices = self._greedy_sample(
                features,
                target,
            )

        elif self.method == "random":

            indices = self._random_sample(
                n,
                target,
            )

        else:

            raise ValueError(
                f"Unknown coreset method: "
                f"{self.method}"
            )

        return features[indices]



class PatchCoreMemory:

    def __init__(self, k=1):

        self.k = k
        self.index = None


    def fit(self, memory):

        memory = np.asarray(
            memory,
            dtype=np.float32
        )

        dimension = memory.shape[1]

        self.index = faiss.IndexFlatL2(
            dimension
        )

        self.index.add(
            memory
        )


    def predict(self, features):

        features = np.asarray(
            features,
            dtype=np.float32
        )

        distances, indices = (
            self.index.search(
                features,
                self.k
            )
        )

        # ----------------------------------------------------
        # k nearest neighbor distance
        # ----------------------------------------------------

        patch_scores = distances[:, 0]

        # ----------------------------------------------------
        # Image-level anomaly score
        # ----------------------------------------------------

        score = float(
            np.max(
                patch_scores
            )
        )

        return (
            score,
            patch_scores
        )