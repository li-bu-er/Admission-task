装依赖：
pip install -r requirements.txt

生成 toy-repo：
python data/download.py
写出 `data/toy-repo/`（含 buggy 快照）。可选 `--with-swebench` 拉 SWE-bench Lite 抽样。

启动本地模型（Ollama 示例）：
set PATH=%LOCALAPPDATA%\Programs\Ollama;%PATH%
ollama pull qwen2.5-coder:7b-instruct
start "ollama" ollama serve
set OPENAI_BASE_URL=http://localhost:11434/v1
set OPENAI_API_KEY=ollama
set OPENAI_MODEL=qwen2.5-coder:7b-instruct


MCP / Skill 自检：
python eval/run.py
`mcp_server_lists_tools`、`skill_loader_metadata` 通过；缺 toy-repo 时 `toy_repo_patch` 跳过。

手动跑 Agent：
python run_agent.py --no-llm
python run_agent.py
`--no-llm` 走确定性兜底修 toy-repo；有 LLM 时走完整 agentic loop。trace 写入 `eval/toy_trace.json`。

跑完整自检：
python eval/run.py
`toy_repo_patch` 要求修好后 `pytest` 通过；`swebench_lite_sample` 未下载则跳过。
