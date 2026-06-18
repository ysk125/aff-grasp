# SegFormer / InternImage 7実験の詳細とqサーバー実行手順

## 1. 実験の目的

Aff-Graspの既存GATモデルとは異なるセマンティックセグメンテーションモデルを用い、backboneの更新方法によってAED性能がどのように変化するか比較した。

比較したモデル系列は次の2種類である。

- SegFormer: MiT-B5 backbone + SegFormer MLP decode head
- InternImage: InternImage-S backbone + UPerNet decode head + auxiliary FCN head

総パラメータ数は、DINOv2-base + GAT headと概ね同程度の約80Mから85Mに揃えた。今回のモデル入力はRGB画像のみであり、既存GATのDepthFeature Injectorやdepth画像は使用していない。

実験コードはMMSegmentationのrunnerを使用せず、`experiments/affgrasp_mmseg/`に実装したプロジェクト固有のPyTorch学習ループで動作する。そのため「MMSegmentation標準の160k iteration recipeを使用した」とは表現しない。

## 2. 対象とした7実験

### SegFormer

| 実験 | backbone更新方法 | 学習対象の概要 |
|---|---|---|
| `segformer_a` | Frozen backbone | MiT-B5 encoderを凍結し、decode headと9クラスclassifierを学習 |
| `segformer_b` | Partial fine-tuning | encoder stage 3・4と対応するpatch embedding、decode headを学習 |
| `segformer_c` | LoRA | encoder stage 3・4のattention `query`・`value`へLoRAを挿入し、decode headとともに学習 |
| `segformer_d` | Full fine-tuning | encoderとdecode headをすべて学習 |

SegFormerは4 stage構成である。コード上のstage indexは0始まりであり、stage 3・4は`block.2`、`block.3`および`patch_embeddings.2`、`patch_embeddings.3`に対応する。

`segformer_c`のLoRA設定は次の通りである。

```text
rank: 8
alpha: 4
dropout: 0.1
target modules: query, value
target stages: stage 3, stage 4
```

LoRAのbase linear層は凍結し、低rank行列とdecode headを学習した。

### InternImage

| 実験 | backbone更新方法 | 学習対象の概要 |
|---|---|---|
| `internimage_a` | Frozen backbone | InternImage-S backboneを凍結し、UPerNetとauxiliary headを学習 |
| `internimage_c` | Adapter | backboneを凍結し、各stage出力後のfeature adapter、UPerNet、auxiliary headを学習 |
| `internimage_d` | Full fine-tuning | backbone、UPerNet、auxiliary headをすべて学習 |

InternImage-Sの主な構成は次の通りである。

```text
core operation: DCNv3
channels: [80, 160, 320, 640]
depths: [4, 4, 21, 4]
groups: [5, 10, 20, 40]
drop path rate: 0.3
normalization: LayerNorm
post norm: enabled
UPerNet channels: 512
auxiliary head channels: 256
auxiliary loss weight: 0.4
PPM scales: [1, 2, 3, 6]
```

`internimage_c`では、各stageのfeature mapへresidual形式の1x1 convolution adapterを追加した。reductionは4である。

## 3. 事前学習重み

### SegFormer

NVIDIAの`segformer-b5-finetuned-ade-640-640`を使用した。qサーバー上では`/opt/models/segformer-b5-ade20k`から読み込む。

ADE20Kの150クラスclassifierは形状が異なるため再利用せず、Aff-Grasp用の9クラスclassifierを新規初期化した。MiT-B5 backboneとdecode headの互換部分にはADE20K事前学習重みを使用した。

### InternImage

公式OpenGVLab InternImage実装と、次のADE20K checkpointを使用した。

```text
/opt/InternImage/checkpoints/upernet_internimage_s_512_160k_ade20k.pth
```

backbone、UPerNet中間層、auxiliary FCN head中間層の互換部分を読み込んだ。ADE20Kの150クラスclassifierは使用せず、main classifierとauxiliary classifierを9クラスで新規初期化した。checkpoint読込時は、classifier以外のmissing keyやunexpected keyがある場合にエラーとする。

## 4. データセットと分割

### 学習・validation

`ego_train`から、RGB画像と`*-label.png`が正しく対になり、対象物体名からアフォーダンスを決定できる331サンプルを使用した。Hugging Face上のファイル総数は画像、label、depth等を合算した数であり、学習サンプル数とは一致しない。

331サンプルを固定seed `1311`で次のように分割する。

```text
train: 281画像（約85%）
validation: 50画像（約15%）
```

validationは`ego_train`内部から作成し、AEDは含めない。best checkpointはvalidation mIoUが最大となったepochで選択する。

### Test

AED全721画像をtest専用として使用した。AEDは学習、validation、checkpoint選択には使用しない。split検証では次を強制している。

- `train.txt`と`val.txt`は`ego_train`配下のみ
- `test.txt`はAED配下のみ

## 5. クラスと教師mask

教師maskと予測は、1枚のclass-index maskとして扱う。複数binary maskではない。

| ID | class |
|---:|---|
| 0 | background |
| 1 | grasp |
| 2 | cut |
| 3 | scoop |
| 4 | pound |
| 5 | support |
| 6 | screw |
| 7 | contain |
| 8 | stick |

学習labelでは、画素値128を`grasp`、画素値255をファイル名から決定した物体固有アフォーダンスへ変換する。それ以外はbackgroundとする。AEDでは配布された`.npy` class-index maskを使用する。

## 6. 入力前処理とaugmentation

全実験で入力解像度を448x448に統一した。

学習時:

1. RGB画像を476x476へbicubic resize
2. maskを476x476へnearest-neighbor resize
3. 448x448をrandom crop
4. 水平反転を確率0.5で適用
5. 垂直反転を確率0.5で適用
6. ImageNet mean/stdで正規化

validation時は476x476へresize後、448x448をcenter cropする。AED test時は画像とmaskを直接448x448へresizeする。

## 7. 学習条件

| 項目 | SegFormer | InternImage |
|---|---:|---:|
| epoch | 15 | 15 |
| batch size | 8 | 4 |
| workers | 2 | 2 |
| optimizer | AdamW | AdamW |
| weight decay | 5e-4 | 5e-4 |
| seed | 0 | 0 |

学習率はparameter groupごとに設定した。

| parameter group | learning rate |
|---|---:|
| pretrained backbone | 6e-5 |
| pretrained decoder / head中間層 | 6e-4 |
| 新規9クラスclassifier | 1e-3 |
| LoRA / Adapter | 1e-3 |

epoch 10と12の終了時に、それぞれ学習率を0.1倍するMultiStepLRを使用した。全体は15 epochであり、「10/12」は全体epoch数ではなくdecay milestoneを意味する。

損失はFocal Lossとmulti-class Dice Lossの和である。

```text
loss = focal_loss(gamma=2, alpha=1) + dice_loss
```

InternImageの学習時のみauxiliary出力を加える。

```text
total_loss = main_loss + 0.4 * auxiliary_loss
```

InternImageのUPerNet pooling branchにはBatchNormがあるため、最後のtrain batchが1サンプルになる場合だけ`drop_last=True`とする。

## 8. パラメータ数

実測したパラメータ数は次の通りである。

| 実験 | 総パラメータ | 学習可能パラメータ | 学習可能率 |
|---|---:|---:|---:|
| `segformer_a` | 84,600,265 | 3,157,257 | 3.73% |
| `segformer_b` | 84,600,265 | 80,774,345 | 95.48% |
| `segformer_c` | 85,059,017 | 3,616,009 | 4.25% |
| `segformer_d` | 84,600,265 | 84,600,265 | 100.00% |
| `internimage_a` | 79,758,386 | 31,255,826 | 39.19% |
| `internimage_c` | 80,031,886 | 31,529,326 | 39.40% |
| `internimage_d` | 79,758,386 | 79,758,386 | 100.00% |

`internimage_a`で学習可能率が約39%あるのは、backboneは凍結している一方、UPerNetとauxiliary headを学習するためである。`segformer_b`はMiT-B5のstage 3が大きいため、後段2 stageのみの更新でも学習可能率が高い。

## 9. 実験結果

`result/affgrasp_full_results/`へ回収した結果を示す。validationはbest checkpoint選択用、testはAED 721画像に対する最終評価である。

| 実験 | best epoch | val mIoU | val F1 | AED mIoU | AED F1 | AED Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| `segformer_a` | 14 | 0.2219 | 0.3195 | 0.1797 | 0.2722 | 0.9067 |
| `segformer_b` | 13 | 0.4710 | 0.6282 | 0.2701 | 0.3941 | 0.9254 |
| `segformer_c` | 15 | 0.3934 | 0.5246 | 0.2742 | 0.4053 | 0.9177 |
| `segformer_d` | 14 | 0.5374 | 0.6911 | 0.2666 | 0.3864 | 0.9227 |
| `internimage_a` | 3 | 0.3139 | 0.4470 | 0.2502 | 0.3783 | 0.9003 |
| `internimage_c` | 15 | 0.3433 | 0.4711 | 0.2336 | 0.3614 | 0.9008 |
| `internimage_d` | 15 | 0.5168 | 0.6682 | **0.3380** | **0.4920** | 0.9197 |

この表のmIoU/F1は既存評価実装に従いbackgroundをmacro平均から除外し、GTに存在する前景classだけを平均している。Accuracyは全valid pixelを対象とするため、background比率の影響を強く受ける。

AED mIoU/F1では`internimage_d`が7実験中で最良だった。一方、全実験でAccuracyが0.90前後あるのに対してmIoU/F1が低い。背景正解によるAccuracy上昇か、前景の位置・class誤りかを切り分けるため、追加診断を実装した。

## 10. 追加した背景・前景診断

各実験について次を保存する。

- background込みmIoU/F1/Accuracy
- background除外mIoU/F1
- GT前景pixelだけのAccuracy
- GT foreground ratio
- predicted foreground ratio
- foreground ratio gap
- 9x9 confusion matrix
- 画像別診断
- foreground過小予測・過大予測Top 20

background除外mIoU/F1は、既存評価と同じくGTに存在するclassだけを平均する。診断実行時も既存の`test/metrics.csv`は上書きしない。

qサーバーで既存best checkpointを再評価する場合:

```bash
cd ~/workspace/aff-grasp
git pull

AFFGRASP_OUTPUT_ROOT=outputs_full_lrfix \
bash scripts/run_affgrasp_diagnostics_detached.sh 1

docker logs -f affgrasp-mmseg-diagnostics
```

主な追加出力:

```text
outputs_full_lrfix/all_experiments_diagnostics_summary.csv

outputs_full_lrfix/<experiment>/test/
  diagnostics_summary.csv
  diagnostics_per_image.csv
  confusion_matrix_raw.csv
  confusion_matrix_normalized_by_gt.csv
  confusion_matrix_raw.png
  confusion_matrix_normalized_by_gt.png
  key_confusions.csv
  foreground_underprediction_cases/
  foreground_overprediction_cases/
```

`key_confusions.csv`には次の混同を抜き出す。

```text
GT grasp -> background
GT cut -> background
GT scoop -> background
GT contain -> scoop
GT screw -> stick
GT stick -> background
```

## 11. qサーバーでの再現手順

### GPUとサーバー状態の確認

```bash
hostname
whoami
nvidia-smi
docker ps
```

他の利用者のprocessがあるGPUは使用しない。7実験は単一GPU上で逐次実行する設計であり、同じGPU上に複数の学習jobを重ねない。

### Docker imageのbuild

InternImageを含めるため、qホスト上でDCNv3対応imageをbuildする。

```bash
cd ~/workspace/aff-grasp
AFFGRASP_WITH_INTERNIMAGE=1 bash scripts/build_docker.sh
```

Docker build中はGPU runtimeを利用できないため、InternImageのDCNv3拡張はcontainer起動後にGPUが見える状態で準備する構成となっている。

### preflight

```bash
bash scripts/run_docker.sh 1
python experiments/affgrasp_mmseg/preflight.py \
  --check-timm-models \
  --check-internimage
exit
```

最低限、次を確認する。

```text
train_samples: 331
aed_samples: 721
transformers_available: true
check_backbone: mit_b5
check_pretrained: true
internimage_backend: official
internimage_model_class: OfficialInternImageSegmentationModel
```

`timm_has_mit_b0`や`timm_has_internimage_t_1k_224`がfalseでも問題ない。SegFormerはTransformers、InternImageは公式OpenGVLab実装を使用する。

### 全7実験のsmoke test

```bash
bash scripts/run_docker.sh 1

AFFGRASP_SMOKE_OUTPUT_ROOT=outputs_smoke_strict \
AFFGRASP_INCLUDE_EXPERIMENTAL_INTERNIMAGE=1 \
bash experiments/affgrasp_mmseg/run_all_smoke_tests.sh 0
```

smoke testは各実験を1 epoch、train 8画像、validation 4画像、AED test 4画像で実行する。モデル生成、事前学習重み、freeze policy、LoRA/Adapter、loss、checkpoint、test推論、診断出力まで確認する。

### 全7実験の本番実行

qホスト側で実行する。

```bash
AFFGRASP_OUTPUT_ROOT=outputs_full_lrfix \
AFFGRASP_EXPERIMENT_FAMILY=all \
AFFGRASP_INCLUDE_EXPERIMENTAL_INTERNIMAGE=1 \
bash scripts/run_all_mmseg_experiments_detached.sh 1 affgrasp-mmseg-all
```

監視:

```bash
docker logs -f affgrasp-mmseg-all
docker ps -a --filter name=affgrasp-mmseg-all
nvidia-smi
```

実行順は次の通りである。

```text
segformer_a
segformer_d
segformer_b
segformer_c
internimage_a
internimage_d
internimage_c
```

## 12. 各実験の保存物

```text
outputs_full_lrfix/<experiment>/
  config.py
  config.yaml
  checkpoints/
    best.pth
  logs/
    history.csv
    parameter_summary.json
  metrics.csv
  visualizations/
  test/
    metrics.csv
    image_manifest.csv
    visualizations/
    worst_cases.csv
    worst_cases/
    diagnostics_summary.csv
    diagnostics_per_image.csv
    confusion_matrix_raw.csv
    confusion_matrix_normalized_by_gt.csv
```

- `logs/history.csv`: 全epochのtrain/validation loss、mIoU、F1、Accuracy
- `logs/parameter_summary.json`: 総パラメータ数、学習可能数、比率
- `metrics.csv`: best validation epochの指標とcheckpoint情報
- `checkpoints/best.pth`: validation mIoU最大時の重み
- `test/metrics.csv`: best checkpointによるAED全件評価
- `test/image_manifest.csv`: AED画像ごとの指標とpanel path
- `test/worst_cases/`: AED mIoU下位例
- `test/diagnostics_*.csv`: 背景・前景を切り分けた追加診断

## 13. 実験解釈上の注意

1. AEDはtest専用であり、モデル選択に使用していない。
2. 7実験はRGBのみで、既存GATのdepth入力とは条件が異なる。
3. Accuracyはbackground優勢の影響を受けるため、mIoU/F1と前景診断を併記する。
4. `segformer_b`と`segformer_d`の差は小さく、後段stage更新だけでも多くのparameterを更新している点に注意する。
5. LoRA/Adapterは学習可能parameterを減らす手法であり、必ずしもfull fine-tuningより高精度になるとは限らない。
6. 今回の結果は331サンプルの小規模学習であり、モデル構造だけでなくデータ量、class不均衡、ラベル品質の影響を含む。
