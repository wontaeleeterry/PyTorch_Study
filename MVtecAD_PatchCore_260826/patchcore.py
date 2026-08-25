import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import faiss

# ============================================================
# 1. Feature Processing (Feature Map -> Patch Embeddings)
# ============================================================

def local_neighborhood_pooling(feature, patch_size=3):
    """
    PatchCore 논문 §3.1의 핵심: Locally aware patch feature (Eq 1~2) 생성.

    목적: 각 픽셀의 특징(feature)을 주변 이웃(patch_size x patch_size)의 평균값으로
    업데이트하여, 공간적 변동(spatial variation)에 강건한(robust) 특징을 추출합니다.

    Args:
        feature: [B, C, H, W] - CNN에서 추출된 특징 맵
        patch_size: 이웃 크기 (논문 권장값 p=3)
    Returns:
        [B, C, H, W] - 평균 풀링이 적용된 특징 맵 (해상도 유지)
    """
    if patch_size is None or patch_size <= 1:
        return feature

    # stride=1, padding=p//2 설정을 통해 출력 해상도를 입력과 동일하게 유지 (Eq 3 대응)
    padding = patch_size // 2
    return F.avg_pool2d(
        feature,
        kernel_size=patch_size,
        stride=1,
        padding=padding,
    )


def feature_map_to_patches(feature):
    """
    2D 특징 맵을 1D 패치 시퀀스로 변환 (Flattening).
    KNN(K-Nearest Neighbor) 연산을 위해 각 공간 위치를 하나의 샘플(Point)로 취급합니다.

    Args:
        feature: [B, C, H, W] - 4D 텐서
    Returns:
        [B, H*W, C] - [Batch, Number of Patches, Channels]
    """
    b, c, h, w = feature.shape
    # [B, C, H, W] -> [B, H, W, C] (채널을 마지막으로 이동)
    feature = feature.permute(0, 2, 3, 1)
    # [B, H, W, C] -> [B, H*W, C] (공간 차원을 하나로 통합)
    feature = feature.reshape(b, h * w, c)
    return feature


def aggregate_features(features, patch_size=3, layers=None):
    """
    PatchCore의 Multi-scale Feature Aggregation.
    서로 다른 해상도를 가진 여러 레이어(기본: layer2, layer3)의 특징을 결합합니다.

    NOTE: 원본 MIMII 코드에는
        layer3 = local_neighborhood_param(layer3, patch_size=patch_size)
    로 정의되지 않은 함수를 호출하는 오타(버그)가 있었다. 아래에서는
    `local_neighborhood_pooling` 으로 수정하고, layer 개수를 2개로 고정하지 않도록
    일반화했다 (config.LAYERS 에 임의 개수의 레이어를 지정할 수 있음).

    Logic:
    1. 각 레이어에 대해 Local Neighborhood Pooling 적용 (Locally aware)
    2. 나머지 레이어들을 첫 번째(가장 고해상도) 레이어의 공간 해상도에 맞춰 Resize
       (Bilinear Interpolation) - 논문 §3.1
    3. 모든 레이어의 특징을 채널(Channel) 방향으로 결합 (Concatenate)

    Args:
        features: {layer_name: [B, C, H, W], ...}  (feature_extractor.extract()의 출력)
        patch_size: 이웃 크기
        layers: 사용할 레이어 이름 순서. None이면 features.keys() 순서를 그대로 사용.
                리스트의 첫 번째 레이어가 기준 해상도(reference resolution)가 된다.
    Returns:
        patches: [B, H_ref*W_ref, C_total] - 결합된 패치 임베딩
        grid_size: (H_ref, W_ref) - 세그멘테이션 맵 복원에 필요한 기준 해상도
    """
    layer_names = list(layers) if layers is not None else list(features.keys())
    if len(layer_names) == 0:
        raise ValueError("최소 한 개 이상의 layer가 필요합니다.")

    # [Step 1] 각 해상도 레벨에서 이웃 정보 집계 (Spatial Robustness 확보)
    pooled = [
        local_neighborhood_pooling(features[name], patch_size=patch_size)
        for name in layer_names
    ]

    # 기준 해상도: 첫 번째(가장 고해상도) 레이어
    ref_h, ref_w = pooled[0].shape[2], pooled[0].shape[3]

    # [Step 2] 기준 해상도 이외의 레이어들을 bilinear로 resize
    resized = [pooled[0]]
    for fmap in pooled[1:]:
        resized.append(
            F.interpolate(
                fmap,
                size=(ref_h, ref_w),
                mode="bilinear",
                align_corners=False,
            )
        )

    # [Step 3] 각 레이어의 2D 맵을 1D 패치 시퀀스로 변환 후 채널 방향 결합
    patch_seqs = [feature_map_to_patches(fmap) for fmap in resized]
    patches = torch.cat(patch_seqs, dim=-1)  # [B, ref_h*ref_w, C_total]

    return patches, (ref_h, ref_w)


# ============================================================
# 2. Coreset Sampling (Memory Bank Compression)
# ============================================================

class CoresetSampler:
    """
    PatchCore §3.2, Algorithm 1: 대규모 메모리 뱅크를 줄이기 위한 근사적
    Greedy Coreset 선택 (minimax facility location, Eq.5).

    핵심 아이디어: 모든 특징을 저장하는 대신, 데이터를 대표하는 핵심 샘플(Coreset)만 남김.
    Greedy k-center 알고리즘을 사용하여, 현재 선택된 샘플들로부터
    가장 멀리 떨어진(가장 이질적인) 샘플을 반복적으로 추가하여 커버리지를 극대화함.
    """
    def __init__(self, ratio=0.10, random_seed=42, method="greedy", projection_dim=128, device=None):
        self.ratio = ratio
        self.random_seed = random_seed
        self.method = method          # 'greedy' 또는 'random'
        self.projection_dim = projection_dim  # 차원 축소 차원 (JL Lemma 활용)
        self.device = device if device is not None else torch.device("cpu")

    def _random_projection(self, features):
        """
        Johnson-Lindenstrauss(JL) Lemma를 이용한 차원 축소.
        고차원 특징을 저차원으로 투영하여 거리 계산(KNN)의 연산 복잡도를 획기적으로 낮춤.
        """
        dim = features.shape[1]
        if self.projection_dim is None or self.projection_dim >= dim:
            return features

        generator = torch.Generator().manual_seed(self.random_seed)
        # 가우시안 분포를 따르는 투영 행렬 생성
        projection_matrix = (
            torch.randn(dim, self.projection_dim, generator=generator)
            / (self.projection_dim ** 0.5)
        ).to(features.device)

        return features @ projection_matrix

    def _greedy_sample(self, features_np, target):
        """
        Greedy k-center 알고리즘 구현.
        1. 첫 번째 중심점은 무작위 선택.
        2. 현재 중심점 집합에서 가장 먼(Max-Min distance) 점을 다음 중심으로 선택.
        """
        n = len(features_np)
        features_t = torch.from_numpy(features_np).to(self.device)

        # 연산 속도를 위해 저차원으로 투영된 공간에서 거리 계산
        proj = self._random_projection(features_t)
        rng = np.random.default_rng(self.random_seed)

        # 초기 중심점 설정
        first_idx = int(rng.integers(0, n))
        selected_indices = [first_idx]

        # 모든 점과 첫 번째 중심점 간의 거리 계산
        min_distances = torch.cdist(proj, proj[first_idx : first_idx + 1]).squeeze(1)
        min_distances[first_idx] = -1.0  # 이미 선택된 점은 제외

        for _ in tqdm(range(1, target), desc="Greedy coreset sampling"):
            # 현재 중심점들 중 가장 멀리 있는 점(Max-Min) 찾기
            next_idx = int(torch.argmax(min_distances).item())
            selected_indices.append(next_idx)

            # 새로 추가된 중심점과 다른 점들 간의 거리 계산
            new_distances = torch.cdist(proj, proj[next_idx : next_idx + 1]).squeeze(1)

            # 각 점마다 '현재까지의 중심점들 중 가장 가까운 거리'를 업데이트
            min_distances = torch.minimum(min_distances, new_distances)
            min_distances[next_idx] = -1.0

        return np.array(selected_indices)

    def sample(self, features):
        """사용자가 요청한 비율(ratio)만큼의 샘플 인덱스를 반환합니다."""
        features = np.asarray(features, dtype=np.float32)
        n = len(features)
        target = max(1, int(n * self.ratio))

        if target >= n:
            return features

        if self.method == "greedy":
            indices = self._greedy_sample(features, target)
        elif self.method == "random":
            indices = np.random.default_rng(self.random_seed).choice(n, size=target, replace=False)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return features[indices]


# ============================================================
# 3. PatchCore Memory & Prediction (Scoring + Segmentation)
# ============================================================

class PatchCoreMemory:
    """
    PatchCore §3.3: 메모리 뱅크 관리 및 이상 점수(Anomaly Score) 계산.

    핵심: 단순 거리(L2)가 아닌, 주변 이웃의 밀도를 고려한 'Reweighted Distance'를 사용.
    MVTec AD는 오디오(MIMII)와 달리 픽셀 레벨 ground-truth mask가 있으므로,
    이미지 레벨 score뿐 아니라 패치 점수를 2D 맵으로 복원해 세그멘테이션도 제공한다.
    """
    def __init__(self, k=1, num_reweight_neighbors=9, gaussian_sigma=4.0):
        self.k = k  # 테스트 패치당 가장 가까운 정상 샘플 수
        self.num_reweight_neighbors = num_reweight_neighbors  # 재가중치 계산용 이웃 수
        self.gaussian_sigma = gaussian_sigma
        self.index = None   # FAISS 인덱스
        self.memory = None  # 정상 샘플 특징 집합

    def fit(self, memory):
        """FAISS IndexFlatL2를 사용하여 메모리 뱅크를 인덱싱합니다."""
        self.memory = np.asarray(memory, dtype=np.float32)
        dimension = self.memory.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.memory)

    def predict(self, features):
        """
        테스트 특징에 대해 이상 점수를 예측합니다.

        Math (Eq 6~7):
        1. m* (Nearest Neighbor): 테스트 패치와 가장 가까운 정상 샘플.
        2. s* (Raw Distance): m*와의 거리.
        3. Reweighting: m*가 정상 데이터 중에서도 흔한 패턴(이웃들이 가까움)이면
           점수를 낮추고, m*가 고립된 패턴(이웃들이 멂)이면 점수를 높인다.

        Args:
            features: [N, D] 한 이미지의 패치 특징들 (N = h*w)
        Returns:
            score: float, 이미지 레벨 anomaly score
            patch_scores: [N] 패치별 raw anomaly score (재가중치 적용 전, 세그멘테이션용)
        """
        features = np.asarray(features, dtype=np.float32)

        # [Step 1] 각 테스트 패치에 대해 가장 가까운 정상 샘플(m*)과 그 거리(s*) 찾기
        # faiss search는 제곱 거리(Squared L2)를 반환하므로 sqrt를 취함
        sq_distances, nn_indices = self.index.search(features, self.k)
        m_star_idx = nn_indices[:, 0]
        s_star = np.sqrt(np.clip(sq_distances[:, 0], a_min=0.0, a_max=None))

        # [Step 2] 재가중치(Reweighting)를 위한 주변 이웃(b) 탐색
        b = min(self.num_reweight_neighbors, len(self.memory) - 1)
        if b <= 0:
            image_score = float(np.max(s_star))
            return image_score, s_star

        # image-level score를 결정하는 patch(argmax) 하나에 대해서만 reweighting 적용
        # (Eq.6~7: s* = max over test patches; reweighting은 그 최댓값 패치에만 적용)
        argmax_idx = int(np.argmax(s_star))
        m_star_vector = self.memory[m_star_idx[argmax_idx] : m_star_idx[argmax_idx] + 1]

        _, neighbor_idx = self.index.search(m_star_vector, b + 1)
        neighbor_idx = neighbor_idx[:, 1:]  # 자기 자신 제외
        neighbor_vectors = self.memory[neighbor_idx[0]]  # [b, D]

        m_test_star = features[argmax_idx]
        dist_to_neighbors = np.linalg.norm(
            neighbor_vectors - m_test_star[None, :], axis=-1
        )

        # [Step 3] Softmax-like Reweighting (Eq 7) 적용
        # w = 1 - exp(s*) / sum(exp(dist_to_neighbors))
        combined = np.concatenate([[s_star[argmax_idx]], dist_to_neighbors])
        max_val = np.max(combined)
        exp_combined = np.exp(combined - max_val)  # log-sum-exp trick (수치 안정성)

        numerator = exp_combined[0]
        denominator = exp_combined[1:].sum()
        weight = 1.0 - numerator / (denominator + 1e-12)

        image_score = float(weight * s_star[argmax_idx])

        # patch_scores(=raw s* per patch)는 재가중치 없이 그대로 반환한다.
        # -> 픽셀 레벨 세그멘테이션은 논문 §3.3에서도 raw distance map을 사용.
        return image_score, s_star

    def segmentation_map(self, patch_scores, grid_size, output_size):
        """
        patch_scores([N]=h*w)를 2D 이상 지도(anomaly map)로 복원한 뒤
        원본 해상도로 업샘플하고 가우시안 스무딩을 적용한다 (§3.3).

        Args:
            patch_scores: [h*w] 패치별 raw anomaly score
            grid_size: (h, w) patch grid 크기 (aggregate_features가 반환한 값)
            output_size: (H, W) 최종 출력(원본 이미지) 해상도
        Returns:
            [H, W] numpy 배열, 값이 클수록 이상 확률이 높음
        """
        h, w = grid_size
        seg = patch_scores.reshape(1, 1, h, w)
        seg_t = torch.from_numpy(seg.astype(np.float32))
        seg_up = F.interpolate(seg_t, size=output_size, mode="bilinear", align_corners=False)
        seg_up = seg_up.squeeze().numpy()
        seg_up = gaussian_filter(seg_up, sigma=self.gaussian_sigma)
        return seg_up
