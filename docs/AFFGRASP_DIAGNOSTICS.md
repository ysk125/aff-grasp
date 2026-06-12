# Aff-Grasp 9-class diagnostics

SegFormer / InternImage の AED test 推論について、背景正解により Accuracy だけが高くなっていないかを確認する診断を追加した。

## 実行

既存の7実験の `best.pth` から AED を再推論する。AED は test のみに使用され、checkpoint 選択には使用されない。

```bash
python tools/run_affgrasp_diagnostics.py \
  --output-root outputs_full_lrfix \
  --gpu 0
```

detached Docker で実行する場合:

```bash
AFFGRASP_OUTPUT_ROOT=outputs_full_lrfix \
bash scripts/run_affgrasp_diagnostics_detached.sh 1

docker logs -f affgrasp-mmseg-diagnostics
```

各実験の `test/metrics.csv` は上書きしない。診断結果は `diagnostics_*.csv`、混同行列CSV/PNG、foreground過小・過大予測caseディレクトリへ追加される。指定された代表的なclass混同は `test/key_confusions.csv` に抜き出される。

## 指標定義

- `miou_with_background`, `f1_with_background`: class 0..8のうち、GTに存在するclassをmacro平均する。
- `miou_without_background`, `f1_without_background`: class 0を除き、GTに存在するclass 1..8をmacro平均する。これは既存 `MetricState` のpresence定義に合わせている。
- `accuracy_with_background`: valid pixel全体のpixel accuracy。
- `foreground_only_accuracy`: `GT != 0` のpixelだけを分母にしたclass accuracy。
- `gt_foreground_ratio`, `predicted_foreground_ratio`: valid pixelに占める前景率。
- `foreground_ratio_gap`: prediction前景率からGT前景率を引いた値。負なら過小予測、正なら過大予測の傾向を示す。
- ignore label `255` およびclass範囲外のGT pixelは全指標から除外する。

全実験比較は `outputs_full_lrfix/all_experiments_diagnostics_summary.csv` に保存される。
