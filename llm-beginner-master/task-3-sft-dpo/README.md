下载基座：
set HF_ENDPOINT = "https://hf-mirror.com"
python data/download.py
落到 `models/Qwen2.5-0.5B`，并打印 SFT / DPO 数据获取提示。

下载 SFT 数据：
huggingface-cli download OpenMOSS-Team/moss-003-sft-data moss-003-sft-no-tools.jsonl.zip --repo-type dataset --local-dir ./data/moss-sft


LoRA / loss mask 自检：
python eval/run.py
`lora_param_count`、`loss_masking` 通过；`sft_vs_base` 会因缺少 `ckpt/sft/` 跳过。

SFT：
python train_sft.py
产物在 `ckpt/sft/`。冒烟可加 `--max_steps 50`。

DPO：
huggingface-cli download hiyouga/DPO-En-Zh-20k --repo-type dataset --local-dir ./data/dpo
python train_dpo.py --data ./data/dpo
产物在 `ckpt/dpo/`（需先有 `ckpt/sft/`）。

对比 base / SFT / DPO：
python src/compare.py
结果写入 `ckpt/compare.md`。

跑完整自检：
python eval/run.py
