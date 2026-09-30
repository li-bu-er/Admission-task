下载数据与模型：
set HF_ENDPOINT = "https://hf-mirror.com"
python data/download.py
会拉 BGE embedding / reranker、`data/kb.pdf`，并校验 `data/gold_qa.jsonl`。
只要 PDF：`python data/download.py --skip-models`。


Chunking 自检：
python eval/run.py
`chunking_sanity` 通过；缺索引时 `nndl_gold_recall_at_10` / `rag_end_to_end` 可能跳过或失败。

建索引：
python build_index.py --chunk_size 512 --overlap 128 --strategy recursive
写入 `data/index/`。

检索指标：
python eval_retrieval.py
目标 **Recall@10 > 0.6**；可加 `--no-rerank` 做消融。

端到端（可选，需本地 LLM）：
ollama pull qwen2.5:7b-instruct
默认请求 `http://localhost:11434/v1`；可用 `RAG_API_BASE` / `RAG_MODEL`。
无 LLM 时 `answer()` 仍返回非空摘录 + sources。

跑完整自检：
python eval/run.py
