设置镜像与编码
set HF_ENDPOINT=https://hf-mirror.com
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

下载预训练模型（任务六已经下载了）
mkdir D:\models
hf download uer/t5-base-chinese-cluecorpussmall --local-dir "D:\models\t5-base-chinese"

截取约 1500 step 的数据后训练：
python -c "open('data/train_fast.tsv','w',encoding='utf-8').writelines(open('data/train.tsv',encoding='utf-8').readlines()[:24000]); open('data/dev_fast.tsv','w',encoding='utf-8').writelines(open('data/dev.tsv',encoding='utf-8').readlines()[:1000])"

训练：
python train.py --pretrained_model "D:\models\t5-base-chinese" --save_dir "checkpoints/t5" --train_path "data/train_fast.tsv" --dev_path "data/dev_fast.tsv" --img_log_dir "logs" --img_log_name "T5-Base-Chinese" --batch_size 16 --max_source_seq_len 128 --max_target_seq_len 32 --learning_rate 1e-4 --num_train_epochs 1 --logging_steps 50 --valid_steps 500 --device "cuda:0"

推理：
python inference.py
