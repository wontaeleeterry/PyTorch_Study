"""
test.py가 생성한 결과를 가시화한다.

입력:
    results/<category>_test_results.csv   (file, label, defect_type, score)
    results/<category>_segmentation.npz   (seg_maps: [N,H,W], gt_masks: [N,H,W])
    -> 두 파일은 test.py에서 동일한 순서(test_items 순회 순서)로 저장되므로
       CSV의 i번째 행과 npz의 i번째 seg_map/gt_mask가 서로 대응한다.

기능:
    1. 이미지 레벨 정상/이상 판정
       evaluate.py와 동일한 F1-최적 threshold를 재계산하여 사용한다
       (score >= threshold 이면 "Abnormal"로 판정).
    2. 결함 위치 가시화 (abnormal로 판정된 경우)
       - 픽셀 단위 anomaly heatmap을 원본 이미지 위에 오버레이
       - Otsu(또는 상위 분위수) 이진화로 이상 영역을 추출해 컨투어 + 바운딩박스로 표시
       - ground-truth mask가 있는 경우 함께 표시하고 IoU를 계산

출력:
    results/<category>_visualizations/{idx:03d}_<defect_type>_<pred>.png  (이미지별 상세)
    results/<category>_visualizations/_contact_sheet.png                 (요약 그리드)
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # 화면 없는 환경(서버/컨테이너)에서도 안전하게 저장하기 위함

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.filters import threshold_otsu
from skimage.measure import find_contours

from config import (
    CATEGORY,
    CATEGORIES,
    RESULT_ROOT,
    RESIZE,
    IMAGE_SIZE,
)
from image import load_display_image
from evaluate import evaluate_image_level


# ------------------------------------------------------------------ #
# 데이터 로딩
# ------------------------------------------------------------------ #

@dataclass
class VizData:
    df: pd.DataFrame
    seg_maps: np.ndarray   # [N, H, W]
    gt_masks: np.ndarray   # [N, H, W]
    threshold: float


def load_visualization_data(category: str) -> VizData:
    csv_path = RESULT_ROOT / f"{category}_test_results.csv"
    seg_path = RESULT_ROOT / f"{category}_segmentation.npz"

    if not csv_path.exists() or not seg_path.exists():
        raise FileNotFoundError(
            f"'{category}' 결과 파일이 없습니다. 먼저 실행하세요:\n"
            f"  python train.py --category {category}\n"
            f"  python test.py  --category {category}"
        )

    df = pd.read_csv(csv_path)
    data = np.load(seg_path)
    seg_maps = data["seg_maps"]
    gt_masks = data["gt_masks"]

    if not (len(df) == len(seg_maps) == len(gt_masks)):
        raise ValueError(
            f"결과 개수가 일치하지 않습니다: csv={len(df)}, "
            f"seg_maps={len(seg_maps)}, gt_masks={len(gt_masks)}. "
            f"test.py를 다시 실행해 결과를 재생성하세요."
        )

    # evaluate.py와 동일한 F1-최적 threshold를 사용해 정상/이상 판정 기준을 통일한다.
    image_result = evaluate_image_level(category)
    threshold = float(image_result["threshold"])

    return VizData(df=df, seg_maps=seg_maps, gt_masks=gt_masks, threshold=threshold)


# ------------------------------------------------------------------ #
# 결함 위치(픽셀 레벨) 유틸리티
# ------------------------------------------------------------------ #

def binarize_segmentation(
    seg_map: np.ndarray,
    method: str = "otsu",
    quantile: float = 0.99,
) -> np.ndarray:
    """anomaly score map을 이진 마스크로 변환해 결함 후보 영역을 추출한다."""
    if seg_map.max() - seg_map.min() < 1e-8:
        return np.zeros_like(seg_map, dtype=bool)

    if method == "otsu":
        try:
            thresh = threshold_otsu(seg_map)
        except ValueError:
            thresh = np.quantile(seg_map, quantile)
    elif method == "quantile":
        thresh = np.quantile(seg_map, quantile)
    else:
        raise ValueError(f"지원하지 않는 method: {method}")

    return seg_map >= thresh


def largest_component_bbox(binary_mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """가장 큰 연결 영역의 바운딩박스 (row_min, col_min, row_max, col_max)를 반환한다."""
    if not binary_mask.any():
        return None

    labeled, num_features = ndimage.label(binary_mask)
    if num_features == 0:
        return None

    sizes = ndimage.sum(binary_mask, labeled, range(1, num_features + 1))
    largest_label = int(np.argmax(sizes)) + 1

    all_slices = ndimage.find_objects(labeled)
    row_slice, col_slice = all_slices[largest_label - 1]
    return row_slice.start, col_slice.start, row_slice.stop, col_slice.stop


def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Optional[float]:
    """gt_mask가 비어있지 않은 경우에만 IoU를 계산한다."""
    gt_bool = gt_mask.astype(bool)
    if not gt_bool.any():
        return None
    pred_bool = pred_mask.astype(bool)
    intersection = np.logical_and(pred_bool, gt_bool).sum()
    union = np.logical_or(pred_bool, gt_bool).sum()
    if union == 0:
        return None
    return float(intersection) / float(union)


def normalize_heatmap(seg_map: np.ndarray) -> np.ndarray:
    """시각적 대비를 위해 이미지별 min-max로 0~1 정규화한다."""
    lo, hi = seg_map.min(), seg_map.max()
    if hi - lo < 1e-8:
        return np.zeros_like(seg_map)
    return (seg_map - lo) / (hi - lo)


def overlay_heatmap(image_rgb: np.ndarray, seg_map: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """원본 이미지 위에 anomaly heatmap(jet colormap)을 alpha-blend한다."""
    norm = normalize_heatmap(seg_map)
    cmap = matplotlib.colormaps["jet"]
    heat_rgba = cmap(norm)  # [H, W, 4], 0~1
    heat_rgb = (heat_rgba[..., :3] * 255).astype(np.uint8)

    base = image_rgb.astype(np.float32)
    heat = heat_rgb.astype(np.float32)
    blended = (1 - alpha) * base + alpha * heat
    return np.clip(blended, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ #
# 단일 이미지 시각화
# ------------------------------------------------------------------ #

def draw_localization_panel(ax, image_rgb, binary_mask, bbox, color="red"):
    ax.imshow(image_rgb)
    for contour in find_contours(binary_mask.astype(float), level=0.5):
        # skimage contour는 (row, col) 순서이므로 plot 시 (col, row)로 뒤집는다.
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color=color)

    if bbox is not None:
        row_min, col_min, row_max, col_max = bbox
        rect = patches.Rectangle(
            (col_min, row_min),
            col_max - col_min,
            row_max - row_min,
            linewidth=2,
            edgecolor="yellow",
            facecolor="none",
        )
        ax.add_patch(rect)

    ax.axis("off")


def visualize_single(
    row: pd.Series,
    seg_map: np.ndarray,
    gt_mask: np.ndarray,
    category: str,
    pixel_threshold_method: str = "otsu",
    pixel_quantile: float = 0.99,
    threshold: float = 0.0,
    save_path: Optional[str] = None,
):
    image_rgb = load_display_image(row["file"], resize=RESIZE, image_size=IMAGE_SIZE)

    gt_label = "Abnormal" if row["label"] == 1 else "Normal"
    pred_label = "Abnormal" if row["score"] >= threshold else "Normal"
    correct = gt_label == pred_label

    has_gt_defect = gt_mask.astype(bool).any()
    n_panels = 4 if has_gt_defect else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(4.2 * n_panels, 4.6))

    # (1) 원본 이미지
    axes[0].imshow(image_rgb)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    # (2) anomaly heatmap 오버레이
    overlay = overlay_heatmap(image_rgb, seg_map)
    axes[1].imshow(overlay)
    axes[1].set_title("Anomaly Heatmap")
    axes[1].axis("off")

    # (3) 결함 위치(컨투어 + 바운딩박스) - Abnormal로 "판정"된 경우에만 표시
    binary_mask = None
    bbox = None
    iou = None
    if pred_label == "Abnormal":
        binary_mask = binarize_segmentation(seg_map, method=pixel_threshold_method, quantile=pixel_quantile)
        bbox = largest_component_bbox(binary_mask)
        iou = compute_iou(binary_mask, gt_mask)
        draw_localization_panel(axes[2], image_rgb, binary_mask, bbox, color="red")
        title = "Defect Localization"
        if iou is not None:
            title += f" (IoU={iou:.2f})"
        axes[2].set_title(title)
    else:
        axes[2].imshow(image_rgb)
        axes[2].set_title("Defect Localization\n(predicted Normal → skipped)")
        axes[2].axis("off")

    # (4) ground-truth mask (있는 경우)
    if has_gt_defect:
        axes[3].imshow(image_rgb)
        axes[3].imshow(gt_mask, cmap="Reds", alpha=0.5)
        axes[3].set_title("Ground Truth Mask")
        axes[3].axis("off")

    status = "OK" if correct else "MISCLASSIFIED"
    fig.suptitle(
        f"[{category}] {row['defect_type']}  |  GT: {gt_label}  |  Pred: {pred_label}  "
        f"|  score={row['score']:.3f} (thr={threshold:.3f})  |  {status}",
        fontsize=11,
        color=("black" if correct else "red"),
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    if save_path is not None:
        fig.savefig(save_path, dpi=120)
        plt.close(fig)
        return None
    return fig


# ------------------------------------------------------------------ #
# 요약 컨택트 시트 (여러 샘플을 한 장에)
# ------------------------------------------------------------------ #

def make_contact_sheet(
    viz: VizData,
    category: str,
    save_path: str,
    samples_per_class: int = 4,
    pixel_threshold_method: str = "otsu",
    pixel_quantile: float = 0.99,
):
    df = viz.df
    normal_idx = df.index[df["label"] == 0].tolist()[:samples_per_class]
    abnormal_idx = df.index[df["label"] == 1].tolist()[:samples_per_class]
    selected = normal_idx + abnormal_idx

    n = len(selected)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 2, figsize=(8, 4 * n))
    if n == 1:
        axes = axes[None, :]

    for row_i, idx in enumerate(selected):
        row = df.loc[idx]
        seg_map = viz.seg_maps[idx]
        gt_mask = viz.gt_masks[idx]

        image_rgb = load_display_image(row["file"], resize=RESIZE, image_size=IMAGE_SIZE)
        gt_label = "Abnormal" if row["label"] == 1 else "Normal"
        pred_label = "Abnormal" if row["score"] >= viz.threshold else "Normal"

        axes[row_i, 0].imshow(image_rgb)
        axes[row_i, 0].set_title(f"{row['defect_type']} | GT={gt_label}")
        axes[row_i, 0].axis("off")

        overlay = overlay_heatmap(image_rgb, seg_map)
        axes[row_i, 1].imshow(overlay)

        if pred_label == "Abnormal":
            binary_mask = binarize_segmentation(seg_map, method=pixel_threshold_method, quantile=pixel_quantile)
            for contour in find_contours(binary_mask.astype(float), level=0.5):
                axes[row_i, 1].plot(contour[:, 1], contour[:, 0], linewidth=1.5, color="lime")

        axes[row_i, 1].set_title(f"Pred={pred_label} | score={row['score']:.3f}")
        axes[row_i, 1].axis("off")

    fig.suptitle(f"[{category}] PatchCore Test Result Summary", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


# ------------------------------------------------------------------ #
# 메인
# ------------------------------------------------------------------ #

def visualize_category(
    category: str,
    output_dir=None,
    max_images: Optional[int] = None,
    pixel_threshold_method: str = "otsu",
    pixel_quantile: float = 0.99,
    make_sheet: bool = True,
    sheet_samples: int = 4,
) -> None:

    viz = load_visualization_data(category)

    out_dir = output_dir or (RESULT_ROOT / f"{category}_visualizations")
    out_dir.mkdir(parents=True, exist_ok=True)

    n_total = len(viz.df)
    n_to_plot = n_total if max_images is None else min(max_images, n_total)

    print(f"\n[{category}] {n_to_plot}/{n_total}개 이미지에 대해 시각화 생성 중 "
          f"(threshold={viz.threshold:.4f}, pixel_threshold={pixel_threshold_method})")

    for idx in range(n_to_plot):
        row = viz.df.iloc[idx]
        seg_map = viz.seg_maps[idx]
        gt_mask = viz.gt_masks[idx]

        pred_label = "abnormal" if row["score"] >= viz.threshold else "normal"
        save_path = out_dir / f"{idx:03d}_{row['defect_type']}_{pred_label}.png"

        visualize_single(
            row=row,
            seg_map=seg_map,
            gt_mask=gt_mask,
            category=category,
            pixel_threshold_method=pixel_threshold_method,
            pixel_quantile=pixel_quantile,
            threshold=viz.threshold,
            save_path=str(save_path),
        )

    if make_sheet:
        sheet_path = out_dir / "_contact_sheet.png"
        make_contact_sheet(
            viz,
            category,
            str(sheet_path),
            samples_per_class=sheet_samples,
            pixel_threshold_method=pixel_threshold_method,
            pixel_quantile=pixel_quantile,
        )
        print(f"요약 이미지 저장: {sheet_path}")

    print(f"개별 시각화 저장 위치: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="PatchCore 테스트 결과 시각화 (MVTec AD)")

    parser.add_argument("--category", type=str, default=CATEGORY)
    parser.add_argument("--all", action="store_true", help="config.CATEGORIES 전체에 대해 시각화 생성")
    parser.add_argument("--max_images", type=int, default=None, help="개별 시각화를 생성할 최대 이미지 수 (기본: 전체)")
    parser.add_argument(
        "--pixel_threshold",
        type=str,
        default="otsu",
        choices=["otsu", "quantile"],
        help="결함 위치(픽셀 레벨) 이진화 방법",
    )
    parser.add_argument("--pixel_quantile", type=float, default=0.99, help="pixel_threshold=quantile일 때 사용할 분위수")
    parser.add_argument("--no_contact_sheet", action="store_true", help="요약 컨택트 시트 생성을 건너뜀")
    parser.add_argument("--sheet_samples", type=int, default=4, help="컨택트 시트에 포함할 클래스별 샘플 수")

    args = parser.parse_args()

    categories = CATEGORIES if args.all else [args.category]

    for category in categories:
        try:
            visualize_category(
                category=category,
                max_images=args.max_images,
                pixel_threshold_method=args.pixel_threshold,
                pixel_quantile=args.pixel_quantile,
                make_sheet=not args.no_contact_sheet,
                sheet_samples=args.sheet_samples,
            )
        except FileNotFoundError as e:
            print(f"[경고] {e}")


if __name__ == "__main__":
    main()
