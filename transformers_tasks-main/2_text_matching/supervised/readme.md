设置镜像与编码
set HF_ENDPOINT=https://hf-mirror.com
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

PointWise（单塔）：
python train_pointwise.py --model "bert-base-chinese" --train_path "data/comment_classify/train.txt" --dev_path "data/comment_classify/dev.txt" --save_dir "checkpoints/comment_classify/pointwise" --img_log_dir "logs/comment_classify" --img_log_name "BERT-PointWise" --batch_size 8 --max_seq_len 128 --valid_steps 50 --logging_steps 10 --num_train_epochs 3 --device "cuda:0"

DSSM（双塔）：
python train_dssm.py --model "bert-base-chinese" --train_path "data/comment_classify/train.txt" --dev_path "data/comment_classify/dev.txt" --save_dir "checkpoints/comment_classify/dssm" --img_log_dir "logs/comment_classify" --img_log_name "BERT-DSSM" --batch_size 8 --max_seq_len 128 --valid_steps 50 --logging_steps 10 --num_train_epochs 3 --device "cuda:0"

Sentence-BERT（双塔）：
python train_sentence_transformer.py --model "bert-base-chinese" --train_path "data/comment_classify/train.txt" --dev_path "data/comment_classify/dev.txt" --save_dir "checkpoints/comment_classify/sentence_transformer" --img_log_dir "logs/comment_classify" --img_log_name "BERT-SentenceTransformer" --batch_size 8 --max_seq_len 128 --valid_steps 50 --logging_steps 10 --num_train_epochs 2 --device "cuda:0"

推理：
python inference.py
