import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm

import faiss


# ============================================================
# Feature map -> patch embedding
# ============================================================

def local_neighborhood_pooling(
    feature,
    patch_size=3,
):
    """
    PatchCore §3.1의 "locally aware patch feature" (Eq 1~2).

    각 위치 (h, w)의 특징을 그 자리 하나만으로 쓰지 않고,
    p x p 이웃(Neighbourhood N_p)에 대해 average pooling을
    적용해 만든다. receptive field가 커져서 작은 공간적
    변화(spatial variation)에 더 강건해진다.

    feature: [B, C, H, W]
    patch_size: 이웃 크기 p (논문 기본값 p=3)

    stride=1, padding=p//2 로 H, W를 그대로 유지한다
    (논문 Eq 3에서 striding s=1인 경우와 동일).
    """

    if patch_size is None or patch_size <= 1:

        # p=1은 이웃 집계가 없는 것과 동일 (원본 그대로 반환)
        return feature

    padding = patch_size // 2

    return F.avg_pool2d(
        feature,
        kernel_size=patch_size,
        stride=1,
        padding=padding,
    )


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
    patch_size=3,
):
    """
    Combine layer2 and layer3, with locally aware patch
    aggregation (§3.1) applied to each hierarchy level
    beforehand.

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
    # Locally aware neighbourhood aggregation (Eq 1~2)
    # 각 hierarchy 레벨 "자기 해상도"에서 이웃을 먼저 집계한다.
    # --------------------------------------------------------

    layer2 = local_neighborhood_pooling(
        layer2,
        patch_size=patch_size,
    )

    layer3 = local_neighborhood_pooling(
        layer3,
        patch_size=patch_size,
    )

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
    """
    PatchCore §3.3.

    predict()는 단순 최근접 거리(raw distance) 대신,
    논문 Eq(6)~(7)의 재가중치(reweighting)를 적용한다:

        m_test,*, m* = argmax_{m_test in P(x_test)} argmin_{m in M} ||m_test - m||_2
        s*           = ||m_test,* - m*||_2

        s = (1 - exp(||m_test,* - m*||) / sum_{m in Nb(m*)} exp(||m_test,* - m||)) * s*

    Nb(m*): memory bank M 안에서 m*의 b-nearest-neighbor.
    m*가 M 안에서 이미 고립된(=드문) 패턴이라면, m*의 이웃들도
    m_test,*로부터 멀 가능성이 크고, 그러면 위 softmax형 비율이
    작아져서 (1 - 비율)이 1에 가까워지고 -> s*가 거의 그대로/더 크게
    반영된다. 반대로 m*가 M 안에서 밀집된(=흔한) 패턴이면 재가중치가
    점수를 깎아, "정상 변동성"을 이상으로 오탐하는 것을 줄여준다.
    """

    def __init__(
        self,
        k=1,
        num_reweight_neighbors=3,
    ):

        self.k = k

        self.num_reweight_neighbors = (
            num_reweight_neighbors
        )

        self.index = None

        self.memory = None


    def fit(self, memory):

        memory = np.asarray(
            memory,
            dtype=np.float32
        )

        self.memory = memory

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

        # ----------------------------------------------------
        # 1) 각 test patch의 최근접 memory 벡터(m*)와 거리(s*)
        #    faiss는 squared L2를 반환하므로 sqrt로 논문의
        #    ||.||_2 (raw L2)와 맞춘다.
        # ----------------------------------------------------

        sq_distances, nn_indices = (
            self.index.search(
                features,
                self.k
            )
        )

        m_star_idx = nn_indices[:, 0]

        s_star = np.sqrt(
            np.clip(
                sq_distances[:, 0],
                a_min=0.0,
                a_max=None,
            )
        )

        # ----------------------------------------------------
        # 2) m*마다, memory bank 안에서 m* 자신의
        #    b-nearest-neighbor (자기 자신 제외)를 찾는다.
        #    같은 m*를 공유하는 test patch가 많을 수 있으므로
        #    중복 없이 한 번씩만 계산한다 (np.unique).
        # ----------------------------------------------------

        b = min(
            self.num_reweight_neighbors,
            len(self.memory) - 1,
        )

        if b <= 0:

            # memory bank가 너무 작아 이웃을 구할 수 없는 경우:
            # 재가중치 없이 raw distance만 반환 (fallback)
            return (
                float(np.max(s_star)),
                s_star,
            )

        unique_idx, inverse = np.unique(
            m_star_idx,
            return_inverse=True,
        )

        m_star_vectors = self.memory[
            unique_idx
        ]

        _, neighbor_idx = (
            self.index.search(
                m_star_vectors,
                b + 1,   # 첫 결과는 자기 자신이므로 +1
            )
        )

        # 자기 자신(첫 번째 열) 제외
        neighbor_idx = neighbor_idx[:, 1:]

        neighbor_vectors = self.memory[
            neighbor_idx
        ]
        # neighbor_vectors: [U, b, D]

        # ----------------------------------------------------
        # 3) test patch -> (자신의 m*의) 각 이웃까지의 거리
        # ----------------------------------------------------

        neighbor_vectors_per_patch = (
            neighbor_vectors[inverse]
        )
        # [N, b, D]

        dist_to_neighbors = np.linalg.norm(
            features[:, None, :]
            - neighbor_vectors_per_patch,
            axis=-1,
        )
        # [N, b]

        # ----------------------------------------------------
        # 4) Eq(7) softmax형 재가중치
        #    overflow 방지를 위해 max값을 빼는 log-sum-exp trick
        #    적용 (softmax 값 자체는 동일하게 유지됨).
        # ----------------------------------------------------

        combined = np.concatenate(
            [
                s_star[:, None],
                dist_to_neighbors,
            ],
            axis=1,
        )
        # [N, 1+b]  (열 0 = m*까지 거리, 열 1~b = 이웃들까지 거리)

        max_val = np.max(
            combined,
            axis=1,
            keepdims=True,
        )

        exp_combined = np.exp(
            combined - max_val
        )

        numerator = exp_combined[:, 0]

        denominator = exp_combined[:, 1:].sum(
            axis=1
        )

        ratio = numerator / (
            denominator
            + 1e-12
        )

        weight = 1.0 - ratio

        patch_scores = (
            weight * s_star
        )

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