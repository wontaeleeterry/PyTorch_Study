import argparse

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_recall_curve,
)

from config import RESULT_ROOT, CATEGORY, CATEGORIES


def evaluate_image_level(category: str):

    result_path = (
        RESULT_ROOT
        / f"{category}_test_results.csv"
    )

    df = pd.read_csv(
        result_path
    )

    y_true = df["label"].values
    scores = df["score"].values

    auc = roc_auc_score(
        y_true,
        scores,
    )

    precision, recall, thresholds = precision_recall_curve(
        y_true,
        scores,
    )

    f1_values = (
        2 * precision * recall
        / (precision + recall + 1e-8)
    )

    best_index = f1_values.argmax()
    best_f1 = f1_values[best_index]

    if best_index < len(thresholds):
        best_threshold = thresholds[best_index]
    else:
        best_threshold = scores.mean()

    predictions = (scores >= best_threshold).astype(int)
    final_f1 = f1_score(y_true, predictions)

    n_misclassified = int(np.sum(predictions != y_true))

    return {
        "category": category,
        "image_auroc": auc,
        "best_f1": final_f1,
        "threshold": best_threshold,
        "misclassified": n_misclassified,
        "n_test": len(y_true),
    }


def evaluate_pixel_level(category: str):

    seg_path = (
        RESULT_ROOT
        / f"{category}_segmentation.npz"
    )

    if not seg_path.exists():
        return None

    data = np.load(seg_path)
    seg_maps = data["seg_maps"]  # [N, H, W]
    gt_masks = data["gt_masks"]  # [N, H, W]

    # 정상 이미지의 마스크는 전부 0이므로 그대로 포함해도 무방 (background로 사용됨)
    y_true = gt_masks.reshape(-1)
    y_score = seg_maps.reshape(-1)

    # 마스크가 전부 0(즉 해당 카테고리 test set에 이상 픽셀이 아예 없는 극단적 경우) 방지
    if y_true.max() == y_true.min():
        return None

    pixel_auroc = roc_auc_score(y_true, y_score)
    return {"category": category, "pixel_auroc": pixel_auroc}


def print_report(rows, key, title):

    print(f"\n{title}")
    print("-" * 60)
    for row in rows:
        print(f"{row['category']:<12s} : {row[key]:.4f}")

    values = [row[key] for row in rows]
    if values:
        print("-" * 60)
        print(f"{'Average':<12s} : {np.mean(values):.4f}")


def main():

    parser = argparse.ArgumentParser(
        description="PatchCore 평가 결과 요약 (MVTec AD)"
    )

    parser.add_argument(
        "--category",
        type=str,
        default=CATEGORY,
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="config.CATEGORIES 전체에 대해 평균 성능(논문 Table 1/2 형식)을 출력",
    )

    args = parser.parse_args()

    categories = CATEGORIES if args.all else [args.category]

    image_rows = []
    pixel_rows = []

    for category in categories:

        try:
            image_result = evaluate_image_level(category)
            image_rows.append(image_result)
        except FileNotFoundError:
            print(f"[경고] '{category}' 결과 CSV가 없습니다. test.py를 먼저 실행하세요.")
            continue

        pixel_result = evaluate_pixel_level(category)
        if pixel_result is not None:
            pixel_rows.append(pixel_result)

    print("\n================================")
    print("MVTec AD / PatchCore (ResNet18)")
    print("================================")

    for row in image_rows:
        print(
            f"[{row['category']}] "
            f"Image AUROC={row['image_auroc']:.4f}  "
            f"Best F1={row['best_f1']:.4f}  "
            f"Misclassified={row['misclassified']}/{row['n_test']}"
        )

    if image_rows:
        avg_auroc = np.mean([r["image_auroc"] for r in image_rows])
        avg_f1 = np.mean([r["best_f1"] for r in image_rows])
        total_mis = sum(r["misclassified"] for r in image_rows)
        total_n = sum(r["n_test"] for r in image_rows)
        print("--------------------------------")
        print(f"Average Image AUROC : {avg_auroc:.4f}")
        print(f"Average Best F1     : {avg_f1:.4f}")
        print(f"Total Misclassified : {total_mis}/{total_n}")

    if pixel_rows:
        print("\n--------------------------------")
        for row in pixel_rows:
            print(f"[{row['category']}] Pixel AUROC={row['pixel_auroc']:.4f}")
        avg_pixel = np.mean([r["pixel_auroc"] for r in pixel_rows])
        print("--------------------------------")
        print(f"Average Pixel AUROC : {avg_pixel:.4f}")

    print("================================")


if __name__ == "__main__":

    main()
