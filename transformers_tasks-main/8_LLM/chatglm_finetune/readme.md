我的电脑现存只有8GB，跑不了ChatGLM-6B，本地未能完整跑通

设置镜像与编码
set HF_ENDPOINT=https://hf-mirror.com
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

环境：
pip install -r requirements.txt

训练（模型路径改为本地 D:/models/chatglm-6b）：
python train.py --train_path data/mixed_train_dataset.jsonl --dev_path data/mixed_dev_dataset.jsonl --use_lora True --lora_rank 8 --batch_size 1 --num_train_epochs 1 --save_freq 500 --learning_rate 3e-5 --logging_steps 50 --max_source_seq_len 256 --max_target_seq_len 128 --save_dir checkpoints/finetune --img_log_dir "log/fintune_log" --img_log_name "ChatGLM Fine-Tune" --quantization_bit 4 --device cuda:0

推理：
python inference.py
