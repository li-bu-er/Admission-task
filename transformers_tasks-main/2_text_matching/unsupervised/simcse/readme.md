设置镜像与编码
set HF_ENDPOINT=https://hf-mirror.com
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

训练：
python train.py --model "bert-base-chinese" --train_path "data/LCQMC/train.txt" --dev_path "data/LCQMC/dev.tsv" --save_dir "checkpoints/LCQMC" --img_log_dir "logs/LCQMC" --img_log_name "BERT-ESimCSE" --learning_rate 1e-5 --dropout 0.3 --batch_size 32 --max_seq_len 64 --valid_steps 400 --logging_steps 50 --num_train_epochs 1 --device "cuda:0"

推理：
python inference.py
