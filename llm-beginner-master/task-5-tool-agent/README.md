生成任务集：
python data/download.py
写出 `data/tasks.json` 与 `data/agent-fixtures/`。

启动本地模型（Ollama 示例）：
set PATH=%LOCALAPPDATA%\Programs\Ollama;%PATH%
ollama pull qwen2.5:7b-instruct
start "ollama" ollama serve
set OPENAI_BASE_URL=http://localhost:11434/v1
set OPENAI_API_KEY=ollama
set OPENAI_MODEL=qwen2.5:7b-instruct


工具自检：
python eval/run.py
`tools_individual` 通过（`wiki` 离线会跳过）；`multi_tool_success_rate` 需 LLM。

跑 Agent / 看 trace：
python run_tasks.py
python run_tasks.py --id 1
python run_tasks.py --inject_error
目标：10 题关键词命中率 **> 60%**（看 `final_answer`，不看自报 `success`）。

跑完整自检：
python eval/run.py
