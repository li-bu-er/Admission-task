设置镜像与编码
set HF_ENDPOINT=https://hf-mirror.com
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

预训练权重已在本地 uie-base-zh/（或自行下载 Pky/uie-base-zh）

训练：
python train.py --pretrained_model "uie-base-zh" --save_dir "checkpoints/DuIE" --train_path "data/DuIE/train.txt" --dev_path "data/DuIE/dev.txt" --img_log_dir "logs/" --img_log_name "UIE-Base" --batch_size 8 --max_seq_len 256 --learning_rate 5e-5 --num_train_epochs 5 --logging_steps 10 --valid_steps 50 --device "cuda:0"

推理：
python inference.py
（按需修改 inference.py 中的 checkpoint 路径，如 checkpoints/DuIE/model_best/）
