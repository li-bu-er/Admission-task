---
name: code-review
description: >
  在修改代码前后做代码审查：检查正确性、边界条件、是否误改测试文件、
  命名与可读性。当你需要 review 某次改动或定位潜在 bug 时加载本 Skill。
---

# Code Review Skill

## 何时使用
- 刚用 `write_file` / `git_apply` 改完代码
- issue 要求「修复 bug」但需确认改动是否完整

## 步骤
1. `git_diff` 查看当前改动
2. `read_file` 阅读被改文件关键函数
3. 确认：**不要修改测试文件**
4. 对照 issue 检查是否覆盖边界情况
5. 给出简短审查结论（通过 / 需再改 + 理由）
