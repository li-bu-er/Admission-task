---
name: pr-description-writer
description: >
  根据 git_diff 与 issue 描述生成简洁的 PR / patch 说明。
  当你已经修好代码并希望总结改动动机、方案与测试结果时加载本 Skill。
---

# PR Description Writer Skill

## 何时使用
- 测试已通过，需要产出 patch 说明或提交摘要
- 需要把 Trace 收成人类可读的变更说明

## 输出模板
```
## Summary
- <一句话说明修了什么>

## Changes
- <文件与关键改动>

## Test plan
- [x] python -m pytest
```
