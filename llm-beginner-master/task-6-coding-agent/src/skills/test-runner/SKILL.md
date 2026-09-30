---
name: test-runner
description: >
  运行 pytest、解读失败输出、定位失败断言与被测函数。
  当你需要验证修复是否生效、或根据测试失败信息排查时加载本 Skill。
---

# Test Runner Skill

## 何时使用
- 修改代码后需要确认 `python -m pytest` 是否全绿
- 测试失败，需要从输出反推该改哪里

## 步骤
1. 调用工具 `run_tests`
2. 若 FAILED：阅读失败断言（Expected / Actual）
3. 用 `search_text` / `read_file` 打开被测实现
4. 最小改动修复后再次 `run_tests`
5. 全绿后可结束任务（done）
