下载数据：
set HF_ENDPOINT = "https://hf-mirror.com"
python data/download.py



attention 自检：
python eval/run.py
`attention_correctness`、`causal_mask` 通过；`classifier_accuracy` 会因缺少 `ckpt/best.pt` 跳过。

训练分类器：
python train.py
默认约 8 epoch；目标 **dev acc ≥ 0.80**。最好权重保存于 `ckpt/best.pt`。

Causal toy LM：
python toy_lm.py


画注意力热图：
python visualize.py
会在 `figures/` 生成正/负/长句至少 3 张图。

跑完整自检：
python eval/run.py

