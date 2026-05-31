# Aff-Grasp AED弱点分析: 実装変更まとめ

## 1. この変更の目的

既存Aff-Graspモデルの弱点を、AED全721画像と学習データ全体から分析できるようにしました。

これまでの推論では、低IoU画像だけを保存して定性的に確認していました。しかし、失敗傾向が低IoU群に集中しているかを検証するには、上位群・中位群を含む全画像の比較が必要です。

今回の変更では、次の5実験を実行するための処理を追加しました。

1. AED画像に対する失敗タイプの手動分類
2. 小領域・細長領域に対する定量評価
3. 境界品質評価
4. 背景False Positive評価
5. 学習データ分布とAED性能の比較

基準モデルの条件は変更していません。DepthFeature Injectorは有効なままです。

## 2. AED全件の分析用export

`tools/run_aff_grasp_eval_maps.py`へ、分析専用の実行モードを追加しました。

```bash
python tools/run_aff_grasp_eval_maps.py \
  --artifact-profile analysis \
  --threshold 0.8
```

このモードでは、低IoU画像だけでなくAED全件を保存します。

```text
analysis/<run_id>/
  run_config.json
  summary.json
  manifest.csv
  label_masks/
    pred/
    gt/
  panels/
  metrics/
  figures/
  review/
```

### 保存されるデータ

| 出力                     | 内容                                                            |
| ------------------------ | --------------------------------------------------------------- |
| `label_masks/pred/*.png` | 予測class-ID mask。背景`0`、affordance class `1..8`             |
| `label_masks/gt/*.png`   | GT class-ID mask。予測maskと同じ形式                            |
| `panels/*.png`           | RGB画像、予測overlay、予測mask、GT maskを並べた確認画像         |
| `manifest.csv`           | 画像名、object、mIoU、foreground IoU、各成果物への相対パス      |
| `run_config.json`        | Git commit、checkpoint SHA-256、推論閾値、データパス、class定義 |

class-ID maskは`uint8` PNGで保存します。全クラス確率の`.npy`は保存しないため、全721件を保存しても容量を抑えられます。

従来の`--save-filter low-iou`は残してあります。少数画像を素早く確認したい場合は、これまで通り利用できます。

## 3. AED自動分析

新しく`tools/analyze_aed_metrics.py`を追加しました。

```bash
python tools/analyze_aed_metrics.py \
  --analysis-root analysis/<run_id>
```

### 小領域・細長領域の評価

GT maskを8-connectivityで連結成分に分割します。同一画像・同一class内で、GT領域と予測領域をHungarian法により一対一対応付けします。

対応する予測領域がないGT領域は、IoU、Recall、F1を`0`として扱います。

領域ごとに以下を保存します。

| 指標                  | 内容                                  |
| --------------------- | ------------------------------------- |
| `area`                | GT領域の画素数                        |
| `area_fraction`       | 画像全体に対する領域面積比            |
| `bbox_aspect_ratio`   | `max(width / height, height / width)` |
| `elongation`          | 主軸長 ÷ 短軸長                       |
| `iou`, `recall`, `f1` | 対応付けた予測領域との評価値          |

面積、bbox aspect ratio、elongationは三分位へ分割し、領域F1の比較図を生成します。

### 境界品質の評価

通常IoUとは別に、以下を計算します。

| 指標                        | 内容                                      |
| --------------------------- | ----------------------------------------- |
| `boundary_iou`              | GT境界帯と予測境界帯のIoU                 |
| `boundary_f1`               | 境界画素のprecisionとrecallから計算したF1 |
| `boundary_neighborhood_iou` | GT境界近傍だけに限定したIoU               |

境界幅は2種類を併記します。

| 設定       | 内容              |
| ---------- | ----------------- |
| `diag2pct` | 画像対角線の2%    |
| `3px`      | 厳格な3ピクセル幅 |

### 背景False Positiveの評価

画像ごとに、GT背景領域に対する誤検出を計算します。

```text
background_fp_rate =
  GT背景上で予測がforegroundとなった画素数 / GT背景画素数

fp_share =
  GT背景上のFalse Positive画素数 / 予測foreground画素数
```

背景の複雑さを自動評価するため、GT背景領域上で以下も計算します。

| 指標                            | 内容                                      |
| ------------------------------- | ----------------------------------------- |
| `background_laplacian_variance` | Laplacian variance                        |
| `background_sobel_edge_density` | Sobel勾配強度が64以上となる背景画素の割合 |

### 自動分析の主な出力

```text
metrics/image_metrics.csv
metrics/region_metrics.csv
metrics/class_metrics.csv
metrics/region_bin_summary.csv
metrics/correlations.json
figures/region_f1_by_shape_tertile.png
figures/boundary_f1_vs_image_miou.png
figures/background_fp_vs_laplacian_variance.png
report.md
```

## 4. ブラインド手動レビュー画面

新しく`tools/build_aed_review_bundle.py`を追加しました。

```bash
python tools/build_aed_review_bundle.py \
  --analysis-root analysis/<run_id>
```

生成後、ローカル環境で以下を実行します。

```bash
cd analysis/<run_id>
python -m http.server 8000
```

ブラウザで`http://localhost:8000/review/`を開くと、全721画像を順番に分類できます。

### ブラインド化

レビュー中は、以下を表示しません。

- mIoU
- 上位群・中位群・下位群
- object名

画像順は固定seedでシャッフルされます。先にスコアを見ることで判断が偏ることを防ぎます。

### 付与できるタグ

```text
miss_small_region
miss_thin_region
boundary_error
background_false_positive
under_segmentation
over_segmentation
class_confusion
no_obvious_failure
other
complex_background
uncertain
note
```

複数タグを同時に付けられます。ブラウザの`localStorage`へ進捗を保存するため、途中で中断しても再開できます。

分類後は`annotations.csv`をexportします。

### 手動タグと自動指標の統合

新しく`tools/merge_aed_review_annotations.py`を追加しました。

```bash
python tools/merge_aed_review_annotations.py \
  --analysis-root analysis/<run_id> \
  --annotations /path/to/annotations.csv
```

mIoUはrank方式でほぼ同数の三分位に分割されます。同じmIoUの画像が複数存在しても、群の件数が大きく偏らないようにしています。

タグごとに以下を出力します。

- 群ごとの件数と比率
- Wilson 95%信頼区間
- 下位群対上位群のrisk ratio
- Fisher exact test
- タグ比率の比較図
- 複雑背景タグの有無による背景FP率の比較
- Mann-Whitney U test
- Cliff's delta

## 5. 学習データ分布分析

新しく`tools/analyze_train_distribution.py`を追加しました。

```bash
python tools/download_aff_grasp_assets.py

python tools/analyze_train_distribution.py \
  --analysis-root analysis/<run_id>
```

### 上流実装と合わせた教師mask変換

学習データのラベル変換は、Aff-Grasp上流実装と同じ規則にしています。

| 元mask値 | 意味                        |
| -------- | --------------------------- |
| `128`    | `grasp`                     |
| `255`    | object固有のtask affordance |

object固有classの例:

| object                        | task affordance |
| ----------------------------- | --------------- |
| `knife`, `scissors`           | `cut`           |
| `spoon`, `ladle`              | `scoop`         |
| `fork`                        | `stick`         |
| `hammer`                      | `pound`         |
| `spatula`, `shovel`, `trowel` | `support`       |
| `screwdriver`                 | `screw`         |
| `pan`, `cup`                  | `contain`       |

### class別の出力

```text
metrics/train_distribution.csv
metrics/train_distribution_summary.json
figures/train_pixels_vs_aed_f1.png
figures/train_instances_vs_aed_f1.png
figures/train_region_area_vs_aed_f1.png
```

class数は8と少ないため、相関は探索的な結果として扱います。因果関係を証明するものではありません。

## 6. Dockerとasset取得の安定化

### CUDA拡張の読込

新しく`scripts/affgrasp_env.sh`を追加しました。

以前は、MSDA CUDA拡張を`site-packages`へinstallしようとして権限エラーが発生しました。今回からは`python setup.py build`で生成した`.so`を直接`PYTHONPATH`へ追加します。

同時に、PyTorch共有ライブラリを`LD_LIBRARY_PATH`へ追加します。

```text
/usr/local/lib/python3.10/dist-packages/torch/lib
```

`scripts/bootstrap_affgrasp.sh`と`scripts/run_docker.sh`は、この共通設定を読み込むように変更しました。

### 壊れたsymlinkの修復

`tools/download_aff_grasp_assets.py`を更新しました。

- 壊れたsymlinkを検出して張り直す
- AEDの実画像、depth、GT `.npy`が各721件以上あるか確認する
- checkpointが存在し、1MiB以上あるか確認する
- 全学習データ取得時に、AED内depthと学習用depthを取り違えない
- asset取得が不完全な場合は、成功扱いにせずエラーを出す

### detached Docker実行

新しく`scripts/run_aed_analysis_detached.sh`を追加しました。

```bash
bash scripts/run_aed_analysis_detached.sh 0
```

このスクリプトは、名前付きcontainer`affgrasp-aed-analysis`をバックグラウンドで起動します。SSH接続を切っても処理は継続します。

container内では順番に以下を実行します。

1. CUDA利用可否の確認
2. MSDA拡張importの確認
3. AED全721件のanalysis export
4. AED自動分析
5. ブラインドレビュー画面の生成

途中で失敗した場合は、その時点でcontainerを停止します。

確認コマンド:

```bash
docker ps -a --filter name=affgrasp-aed-analysis
docker logs -f affgrasp-aed-analysis
```

## 7. 追加した依存関係

`requirements-affgrasp.txt`へ`scipy`を追加しました。

用途:

- Hungarian法による領域対応付け
- Spearman相関
- Fisher exact test
- Mann-Whitney U test
- 連結成分解析

Docker imageを再buildしてください。

```bash
bash scripts/build_docker.sh
```

## 8. テスト

以下を追加しました。

```text
tests/test_analysis_common.py
tests/test_analysis_workflow.py
```

確認している内容:

- IoU、Recall、F1
- 未対応GT領域を`0`として扱うこと
- 細長領域のaspect ratio
- 同一maskのBoundary IoUとBoundary F1
- 上流規則に合わせた教師mask変換
- rank方式によるmIoU三分位
- 壊れたasset symlinkの修復
- 疑似AEDによる自動分析、レビュー画面、CSV統合、学習分布分析の一連処理

ローカル検証結果:

```text
8 tests passed
python -m py_compile: passed
bash -n scripts/*.sh: passed
git diff --check: passed
```

## 9. mサーバーで次に行うこと

GitHubへ変更を反映した後、mサーバーで以下を実行します。

```bash
ssh m
cd ~/workspace/aff-grasp
git pull
bash scripts/build_docker.sh
bash scripts/run_docker.sh 0
```

container内でbootstrapと3画像smoke testを行います。

```bash
bash scripts/bootstrap_affgrasp.sh

python tools/run_aff_grasp_eval_maps.py \
  --artifact-profile analysis \
  --threshold 0.8 \
  --max-samples 3

exit
```

問題がなければ、mサーバー側で全件処理を開始します。

```bash
bash scripts/run_aed_analysis_detached.sh 0
docker logs -f affgrasp-aed-analysis
```

## 10. 参考資料

- [Aff-Grasp upstream repository](https://github.com/Reagan1311/Aff-Grasp)
- [Aff-Grasp dataset loader](https://github.com/Reagan1311/Aff-Grasp/blob/main/affordance-learning/data/ego_video_data.py)
- [Boundary IoU paper](https://arxiv.org/abs/2103.16562)
- [AED Hugging Face dataset](https://huggingface.co/datasets/Gen1113/Data_for_Aff-Grasp)
