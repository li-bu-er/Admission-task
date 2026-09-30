下载数据：
set HF_ENDPOINT = "https://hf-mirror.com"
python data/download.py
默认唐诗 quick-start；也可 `--dataset tinystories` / `--dataset skypile`。


Tokenizer / KV cache 自检：
python eval/run.py
`tokenizer_roundtrip`、`kv_cache_equivalence` 在缺 `ckpt/` 时会跳过；`perplexity_on_dev` 同理。

训练：
python train.py
会写 `ckpt/tokenizer.json` 与 `ckpt/best.pt`。唐诗档目标 **dev PPL < 50**（TinyStories 默认 < 10）。

生成样例：
python generate.py --prompt "床前明月光"
对比 greedy / temperature / top-k / top-p。

跑完整自检：
python eval/run.py
