코드 전체를 논문의 방법론(§3.1~3.3)과 하나씩 대조해서 확인해보겠습니다.논문의 3개 핵심 구성요소(§3.1 locally aware patch features, §3.2 coreset, §3.3 anomaly scoring)를 기준으로 코드를 하나씩 대조해봤습니다. 결론부터 말하면, **coreset 부분(§3.2)은 지난번 수정으로 논문과 거의 일치**하지만, **§3.1과 §3.3에 논문의 핵심 아이디어 중 두 가지가 빠져 있습니다.**

## 1. [중요] "Locally Aware" 패치 특징이 실제로는 구현되어 있지 않음 (§3.1)

논문에서 "locally aware patch feature"란, 각 위치 $(h,w)$의 특징을 그 위치 하나만 보고 쓰는 게 아니라, **주변 $p \times p$ 이웃의 특징들을 adaptive average pooling으로 뭉쳐서** 만드는 것입니다 (논문 Eq. 1~2). 이렇게 하면 receptive field가 커지고, 작은 공간적 변화(spatial variation)에 강건해집니다. 논문은 이 이웃 크기 $p$가 성능에 큰 영향을 준다는 것을 보였고(Figure 4 상단), $p=3$을 기본값으로 씁니다.

**현재 `patchcore.py`의 `feature_map_to_patches()`를 보면:**

```python
def feature_map_to_patches(feature):
    b, c, h, w = feature.shape
    feature = feature.permute(0, 2, 3, 1)
    feature = feature.reshape(b, h * w, c)
    return feature
```

이건 단순히 `[B,C,H,W]`를 `[B,H*W,C]`로 reshape만 할 뿐, **이웃 픽셀을 pooling하는 과정이 전혀 없습니다.** 즉 지금 코드는 "locally aware"가 아니라 그냥 "per-pixel" 패치 특징입니다. `feature_extractor.py`, `patchcore.py` 어디에도 `nn.AvgPool2d`나 `F.avg_pool2d` 같은 이웃 집계 연산이 없습니다.

**왜 중요한가**: 논문에서 $p=1$(이웃 집계 없음)일 때 detection AUROC가 가장 낮았고, $p=3$에서 최고점을 찍었습니다(Figure 4). MIMII 오디오 스펙트로그램에서도 이 부분을 넣으면 국소적인 소음 패턴(짧은 튐, 특정 주파수대의 미세한 변화)에 대한 강건성이 좋아질 가능성이 큽니다.

**보완 방향**: `layer2`, `layer3`를 뽑은 직후, patch화 이전에 각각에 `p×p` adaptive average pooling(stride=1, padding으로 크기 유지)을 적용해야 합니다.

## 2. [중요] Anomaly Score의 재가중치(reweighting) 단계가 없음 (§3.3)

논문은 이미지 레벨 anomaly score를 그냥 "가장 가까운 memory bank 벡터까지의 거리"로 쓰지 않습니다. 대신 Eq. 6~7에서:

1. 테스트 패치 중 memory bank와 가장 먼 패치 $m^{test,*}$와, 그 패치의 최근접 memory 벡터 $m^*$를 찾고
2. **$m^*$ 자신이 memory bank 안의 다른 벡터들로부터 얼마나 고립되어 있는지**(= $m^*$의 $b$개 최근접 이웃까지의 거리에 대한 softmax)를 계산해서
3. $m^*$가 "memory bank 안에서도 흔치 않은 패턴"일수록 원래 거리 $s^*$에 가중치를 더 곱해 점수를 올립니다.

논문은 이 재가중치가 단순 최대 거리보다 더 견고(robust)하다고 명시적으로 밝히고 있습니다.

**현재 `PatchCoreMemory.predict()`:**

```python
patch_scores = distances[:, 0]
score = float(np.max(patch_scores))
```

그냥 `raw distance`의 최댓값을 그대로 image-level score로 씁니다. 재가중치 단계가 완전히 빠져있습니다.

**왜 중요한가**: 재가중치가 없으면, memory bank 안에서 흔한 패턴(주변에 비슷한 벡터가 많은)과 드문 패턴(주변에 비슷한 벡터가 거의 없는, 즉 "정상이지만 특이 케이스")이 test patch와 거리가 같을 때 똑같은 점수를 받습니다. 이렇게 되면 실제로는 정상 변동성(nominal variance)일 뿐인 케이스를 이상으로 오탐할 위험이 커집니다 — 이건 앞서 논의한 greedy coreset의 취지(드문 정상 패턴도 잘 대표하자)와도 맞닿아 있는 부분이라, 지금 상태로는 coreset을 아무리 잘 만들어도 scoring 단계에서 그 이점을 온전히 못 살리고 있는 셈입니다.

**보완 방향**: `PatchCoreMemory`에 memory bank 자기 자신에 대한 self-kNN 인덱스를 하나 더 만들어서, 가장 가까운 memory 벡터 $m^*$의 $b$-nearest-neighbor 거리들을 조회하고 Eq. 7의 softmax 가중치를 곱하도록 `predict()`를 수정해야 합니다.

## 3. [참고사항] 백본(backbone) 선택 차이

논문은 기본적으로 **WideResNet50**의 hierarchy 2+3을 씁니다. 현재 코드(`feature_extractor.py`)는 **ResNet18**의 layer2+3을 씁니다. 이건 틀렸다기보다는 계산량을 줄이기 위한 의도적 단순화로 보이지만, 참고로:

- 논문의 backbone ablation(Supplementary Table S6)에서 ResNet50 계열들 간 성능 차이는 크지 않았지만(±1%pt 이내), 다만 ResNet18은 이 표에 아예 없을 만큼 더 얕고 채널 수도 적은 모델이라, receptive field와 표현력 면에서 논문 벤치마크 수치를 그대로 재현하긴 어려울 수 있습니다.
- MIMII는 이미지가 아니라 mel-spectrogram이라는 점을 감안하면, 오히려 ImageNet bias가 덜한 얕은 backbone이 더 유리할 수도 있어 이 자체가 "틀린 선택"은 아닙니다. 다만 이 트레이드오프를 인지하고 선택한 것인지 확인이 필요합니다.

## 4. [경미] Segmentation map에 Gaussian smoothing 없음

`visualize_anomaly.py`에는 patch score를 2D로 재배열하고 리사이즈하는 로직은 있지만, 논문이 언급한 **$\sigma=4$의 Gaussian smoothing**(§3.3 마지막 문장)이 적용되어 있지 않습니다. 이건 성능보다는 시각화 품질(노이즈 완화)에 관한 것이라 우선순위는 낮습니다.

---

## 정리 및 우선순위

| 항목 | 논문 근거 | 현재 상태 | 우선순위 |
|---|---|---|---|
| Locally aware 패치 (이웃 pooling, $p$) | §3.1, Eq 1-3, Fig 4 | ❌ 없음 (단순 reshape) | **높음** |
| Anomaly score 재가중치 | §3.3, Eq 6-7 | ❌ 없음 (raw max distance) | **높음** |
| Greedy coreset + JL projection | §3.2, Algorithm 1 | ✅ 구현됨 | - |
| Backbone (WideResNet50 vs ResNet18) | §4.1, Table S6 | ⚠️ 다름 (의도적 단순화로 추정) | 낮음 (트레이드오프 인지 필요) |
| Segmentation map Gaussian smoothing | §3.3 마지막 문단 | ❌ 없음 | 낮음 |

가장 임팩트가 클 것으로 보이는 두 가지(**locally aware 패치 특징**, **anomaly score 재가중치**)부터 코드에 반영해드릴까요? 순서대로 하나씩 구현해드리겠습니다.