# Aff-Grasp 追実験まとめ

## 概要

本実験では，既存GATの代替として，SegFormerおよびInternImageをAff-Grasp/AED向けの9クラスsemantic segmentation modelとして学習・評価した。
評価はAED test set 721枚に対して行い，指標は mIoU，F1，Accuracy を用いた。

クラス定義は以下の9クラスである。

```text
0: background
1: grasp
2: cut
3: scoop
4: pound
5: support
6: screw
7: contain
8: stick
```

全体傾向として，Accuracyは全実験で約90%以上と高い一方，mIoU/F1は元論文GATより低い結果となった。これは，backgroundを明示クラスとして含む9クラスsemantic segmentationでは，背景画素の正解によりAccuracyが高く出やすい一方で，アフォーダンス前景領域の検出・分類が十分でない可能性を示している。

## 実験結果一覧

| Experiment | Model | Condition | mIoU | F1 | Accuracy | Total Params | Trainable Params | Trainable Ratio | mIoU Rank | F1 Rank |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| segformer_a | SegFormer | A: frozen encoder | 17.97% | 27.22% | 90.67% | 84,600,265 | 3,157,257 | 3.73% | 7 | 7 |
| segformer_b | SegFormer | B: partial fine-tuning | 27.01% | 39.41% | 92.54% | 84,600,265 | 80,774,345 | 95.48% | 3 | 3 |
| segformer_c | SegFormer | C: LoRA fine-tuning | 27.42% | 40.53% | 91.77% | 85,059,017 | 3,616,009 | 4.25% | 2 | 2 |
| segformer_d | SegFormer | D: full fine-tuning | 26.66% | 38.64% | 92.27% | 84,600,265 | 84,600,265 | 100.00% | 4 | 4 |
| internimage_a | InternImage | A: frozen backbone | 25.02% | 37.83% | 90.03% | 79,758,386 | 31,255,826 | 39.19% | 5 | 5 |
| internimage_c | InternImage | C: adapter-based fine-tuning | 23.36% | 36.14% | 90.08% | 80,031,886 | 31,529,326 | 39.40% | 6 | 6 |
| internimage_d | InternImage | D: full fine-tuning | 33.80% | 49.20% | 91.97% | 79,758,386 | 79,758,386 | 100.00% | 1 | 1 |

## 各実験の簡単なまとめ

### 1. segformer_a: SegFormer A / Frozen encoder

SegFormer encoderを凍結し，decode head/classifierのみを学習した条件である。

- mIoU: 17.97%
- F1: 27.22%
- Accuracy: 90.67%
- Trainable ratio: 3.73%
- mIoU/F1 rank: 7位 / 7位

7実験中で最も低いmIoU/F1となった。事前学習済みSegFormer特徴を固定したままheadのみを学習するだけでは，Aff-Grasp/AEDのアフォーダンス領域に十分適応できなかったと考えられる。一方でAccuracyは90%を超えており，背景画素の正解により高く見えている可能性がある。

### 2. segformer_b: SegFormer B / Partial fine-tuning

SegFormerのstage 1,2を凍結し，stage 3,4とdecode head/classifierを学習した条件である。

- mIoU: 27.01%
- F1: 39.41%
- Accuracy: 92.54%
- Trainable ratio: 95.48%
- mIoU/F1 rank: 3位 / 3位

SegFormer-Aから大きく改善し，後段stageを学習することでアフォーダンス領域への適応が進んだ。ただし，trainable ratioが95%以上と高く，実質的にはfull fine-tuningに近い規模の学習である。Accuracyは全実験中で最も高いが，mIoU/F1はInternImage-DやSegFormer-Cに及ばなかった。

### 3. segformer_c: SegFormer C / LoRA fine-tuning

SegFormerのpretrained encoder本体を凍結し，stage 3,4のattention query/valueにLoRAを挿入して学習した条件である。

- mIoU: 27.42%
- F1: 40.53%
- Accuracy: 91.77%
- Trainable ratio: 4.25%
- mIoU/F1 rank: 2位 / 2位

SegFormer系では最良の結果となった。学習可能パラメータは約4.25%と少ないにもかかわらず，SegFormer-B/Dを上回っているため，今回のデータ量ではfull fine-tuningよりもLoRAによるparameter-efficient fine-tuningの方が安定している可能性がある。卒研の提案としては最も扱いやすいSegFormer条件である。

### 4. segformer_d: SegFormer D / Full fine-tuning

SegFormer encoder全体とdecode head/classifierをすべて学習した条件である。

- mIoU: 26.66%
- F1: 38.64%
- Accuracy: 92.27%
- Trainable ratio: 100.00%
- mIoU/F1 rank: 4位 / 4位

SegFormer-Aよりは大きく改善したが，SegFormer-Cを下回った。全パラメータを更新しても性能が最大にならなかったことから，AEDに対する汎化や小規模データでの過学習が影響している可能性がある。単純なfull fine-tuningが常に最良ではないことを示す結果である。

### 5. internimage_a: InternImage A / Frozen backbone

InternImage backboneを凍結し，UPerHead/classifier/auxiliary headを学習した条件である。

- mIoU: 25.02%
- F1: 37.83%
- Accuracy: 90.03%
- Trainable ratio: 39.19%
- mIoU/F1 rank: 5位 / 5位

Frozen条件ではあるが，UPerHeadなどのhead側パラメータが大きいため，trainable ratioは約39%と比較的高い。SegFormer-Aよりは明確に良い結果であり，InternImageの事前学習特徴とUPerHeadの組み合わせはhead-onlyでも一定の適応力を持つと考えられる。ただし，full fine-tuningには大きく劣る。

### 6. internimage_c: InternImage C / Adapter-based fine-tuning

InternImage backbone本体を凍結し，各stage出力後にadapterを追加して学習した条件である。

- mIoU: 23.36%
- F1: 36.14%
- Accuracy: 90.08%
- Trainable ratio: 39.40%
- mIoU/F1 rank: 6位 / 6位

InternImage-Aよりも低い結果となった。今回のadapter設計または学習設定では，InternImage特徴をAff-Grasp/AED向けに有効に補正できなかった可能性がある。adapter追加による性能改善は確認できず，現時点ではInternImageに対するPEFT手法としては再設計が必要である。

### 7. internimage_d: InternImage D / Full fine-tuning

InternImage backbone全体とUPerHead/classifier/auxiliary headをすべて学習した条件である。

- mIoU: 33.80%
- F1: 49.20%
- Accuracy: 91.97%
- Trainable ratio: 100.00%
- mIoU/F1 rank: 1位 / 1位

7実験中で最も高いmIoU/F1を達成した。InternImageのdeformable convolution系backboneを全体fine-tuningすることで，アフォーダンス領域への適応が最も進んだと考えられる。特に，局所形状や細長い領域への適応という観点では有望な結果である。ただし，AccuracyはSegFormer-B/Dと同程度であり，元論文GATのmIoU/F1には届いていないため，背景優勢の評価・9クラスsoftmax設計・前景検出率の追加診断が必要である。

## 実験間の比較

### SegFormer系の傾向

SegFormerでは，Frozen encoderのsegformer_aが最も低く，LoRA fine-tuningのsegformer_cが最も良い結果となった。

```text
segformer_c > segformer_b > segformer_d > segformer_a
```

少量の追加パラメータでstage 3,4を適応させるLoRAが，full fine-tuningよりも良い結果を示した点が重要である。これは，データ量が限られるAff-Grasp/AED設定では，pretrained特徴を大きく崩さずに後段意味特徴だけを補正する方が有効である可能性を示している。

### InternImage系の傾向

InternImageでは，full fine-tuningのinternimage_dが最良であり，adapter-based fine-tuningのinternimage_cはhead-onlyのinternimage_aを下回った。

```text
internimage_d > internimage_a > internimage_c
```

InternImageはbackbone全体を学習した場合に最も性能が伸びた。一方で，今回のadapter設計では改善が見られなかったため，adapterの挿入位置，縮小率，学習率，正規化，headとの接続方法を再検討する必要がある。

### 全体順位

mIoU/F1の順位は以下である。

```text
1. internimage_d
2. segformer_c
3. segformer_b
4. segformer_d
5. internimage_a
6. internimage_c
7. segformer_a
```

今回の7実験では，InternImage-Dが最も高性能であり，SegFormer-Cがparameter-efficientな条件として最も有望であった。

## 現時点での解釈

今回の結果から，以下のことが考えられる。

1. AccuracyだけではAff-Grasp/AEDの性能を正しく評価できない可能性が高い。
2. 9クラスsemantic segmentationではbackground画素の影響によりAccuracyが高く出やすい。
3. SegFormerではLoRA fine-tuningがfull fine-tuningよりも有効だった。
4. InternImageではfull fine-tuningが最も有効だった。
5. InternImage adapterは今回の設計では有効性が確認できなかった。
6. 元論文GATとの差を議論するには，background除外mIoU/F1，foreground-only Accuracy，foreground ratio，confusion matrixによる追加診断が必要である。

## 次に確認すべきこと

次の段階では，以下の診断指標を追加して，Accuracyが高くmIoU/F1が低い原因を切り分ける。

- background込み mIoU/F1/Accuracy
- background除外 mIoU/F1
- foreground pixelだけのAccuracy
- GT foreground ratio
- predicted foreground ratio
- confusion matrix

特に，`GT foreground ratio` と `predicted foreground ratio` を比較することで，モデルがforegroundを出しすぎているのか，あるいはbackgroundに吸われてforegroundを出せていないのかを確認する。

その結果，9クラスbackgroundありのsoftmax設計が問題であると判断される場合は，元論文GATに近い8クラス出力 + sigmoid + threshold background方式を代表モデルで試す。
