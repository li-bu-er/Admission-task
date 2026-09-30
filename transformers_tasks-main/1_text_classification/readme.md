设置镜像与编码
set HF_ENDPOINT=https://hf-mirror.com
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

训练：
python train.py --model "bert-base-chinese" --train_path "data/comment_classify/train.txt" --dev_path "data/comment_classify/dev.txt" --save_dir "checkpoints/comment_classify" --img_log_dir "logs/comment_classify" --img_log_name "BERT" --num_labels 8 --batch_size 16 --max_seq_len 128 --valid_steps 50 --logging_steps 10 --num_train_epochs 5 --device "cuda:0"

推理：
python inference.py
