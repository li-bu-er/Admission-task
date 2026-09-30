设置镜像与编码
set HF_ENDPOINT=https://hf-mirror.com
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

1）训练奖励模型（Reward Model）
python train_reward_model.py --model "bert-base-chinese" --train_path "data/reward_datasets/sentiment_analysis/train.tsv" --dev_path "data/reward_datasets/sentiment_analysis/dev.tsv" --save_dir "checkpoints/reward_model/sentiment_analysis" --img_log_dir "logs/reward_model/sentiment_analysis" --img_log_name "BERT Reward Model" --batch_size 16 --max_seq_len 128 --learning_rate 1e-5 --valid_steps 100 --logging_steps 10 --num_train_epochs 2 --device "cuda:0"

推理：
python inference_reward_model.py

2）PPO（情感模型打分，No Human Reward）
先下载：
hf download uer/gpt2-chinese-cluecorpussmall --local-dir "D:\models\gpt2-chinese"
hf download uer/roberta-base-finetuned-jd-binary-chinese --local-dir "D:\models\roberta-jd-sentiment"

按需在 ppo_sentiment_example.py 中把 batch_size 调小（8GB 显存建议 8~16），再运行：
python ppo_sentiment_example.py

推理：
python inference_ppo_gpt.py
（修改其中 model_path 为实际 checkpoints/ppo_sentiment_gpt/... 目录）
