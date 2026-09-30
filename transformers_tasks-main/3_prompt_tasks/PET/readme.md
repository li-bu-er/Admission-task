设置镜像与编码
set HF_ENDPOINT=https://hf-mirror.com
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

训练：
python pet.py --model "bert-base-chinese" --train_path "data/comment_classify/train.txt" --dev_path "data/comment_classify/dev.txt" --save_dir "checkpoints/comment_classify/" --img_log_dir "logs/comment_classify" --img_log_name "BERT-PET" --verbalizer "data/comment_classify/verbalizer.txt" --prompt_file "data/comment_classify/prompt.txt" --batch_size 8 --max_seq_len 256 --valid_steps 40 --logging_steps 5 --num_train_epochs 30 --max_label_len 2 --rdrop_coef 5e-2 --device "cuda:0"

推理：
python inference.py
