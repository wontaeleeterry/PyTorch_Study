import numpy as np
import torch
import torch.nn.functional as F

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

    def __init__(
        self,
        ratio=0.01,
        random_seed=42,
    ):

        self.ratio = ratio

        self.random_seed = (
            random_seed
        )

    def sample(
        self,
        features,
    ):
        """
        Approximate coreset.

        For the first runnable experiment,
        random sampling is used.

        Later this can be replaced with
        PatchCore's approximate greedy
        coreset algorithm.
        """

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

        rng = np.random.default_rng(
            self.random_seed
        )

        indices = rng.choice(
            n,
            size=target,
            replace=False,
        )

        return features[indices]


# ============================================================
# FAISS Memory Bank
# ============================================================

class PatchCoreMemory:

    def __init__(
        self,
        k=1,
    ):

        self.k = k

        self.index = None

        self.memory = None

    def fit(
        self,
        features,
    ):
        """
        Build FAISS memory bank.
        """

        features = np.asarray(
            features,
            dtype=np.float32,
        )

        self.memory = features

        dimension = (
            features.shape[1]
        )

        self.index = (
            faiss.IndexFlatL2(
                dimension
            )
        )

        self.index.add(
            features
        )

    def predict_patches(
        self,
        features,
    ):
        """
        Return nearest-neighbor
        distance for every patch.
        """

        features = np.asarray(
            features,
            dtype=np.float32,
        )

        distances, _ = (
            self.index.search(
                features,
                self.k,
            )
        )

        return distances[:, 0]

    def predict(
        self,
        features,
    ):
        """
        Patch-level distances ->
        file-level anomaly score.

        Max patch distance is used.
        """

        patch_scores = (
            self.predict_patches(
                features
            )
        )

        score = float(
            np.max(patch_scores)
        )

        return score, patch_scores