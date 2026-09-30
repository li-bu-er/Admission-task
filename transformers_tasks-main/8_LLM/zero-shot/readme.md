备注：
ChatGLM-6B FP16 约需 13GB 显存。本机 RTX 5060 仅 8GB，显存不够，本地未能完整跑通本目录实验。

设置镜像与编码
set HF_ENDPOINT=https://hf-mirror.com
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

环境（建议 llm_env）：
pip install -r requirements.txt
pip install sentencepiece accelerate

下载模型：
hf download THUDM/chatglm-6b --local-dir "D:\models\chatglm-6b"

将脚本中的模型路径改为本地 D:/models/chatglm-6b 后运行：
python llm_classification.py
python llm_text_matching.py
python llm_information_extraction.py

可选调试台：
streamlit run playground.py --server.port 8001
