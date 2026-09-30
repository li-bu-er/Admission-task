备注：
本任务基于 ChatGLM-6B。完整 FP16 加载约需 13GB 显存；本机为 RTX 5060 8GB，显存不足，未能在本地完整跑通 zero-shot / LoRA 微调。
若有 >=16GB 显存机器，可按 zero-shot/readme.md 与 chatglm_finetune/readme.md 运行。

设置镜像与编码
set HF_ENDPOINT=https://hf-mirror.com
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

建议新建独立环境（勿与 graph_test 混用）：
conda create -n llm_env python=3.10 -y
conda activate llm_env
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.30.2 sentencepiece accelerate cpm_kernels

下载模型：
mkdir D:\models
hf download THUDM/chatglm-6b --local-dir "D:\models\chatglm-6b"

zero-shot 示例（需先把脚本中的模型路径改为 D:/models/chatglm-6b）：
cd zero-shot
python llm_classification.py
python llm_text_matching.py
python llm_information_extraction.py
