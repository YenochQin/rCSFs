# Gd I CSFG 批量分层验证（2026-10-30）

目的：检验新批量配置能否生成 AS0 零激发和 AS1–AS4 SD 完整空间，
并与原版 GRASPG 对 AS4 的拆分保持文件一致。判据为逐字节一致，
不是仅比较数量；不涉及 SCF/RCI 或物理收敛结论。

输入见 [完整 TOML](../../examples/gd_csfg_batch.toml)，线程数 8。
运行当前源码 CLI；全部五层生成墙钟时间 4.070 秒（单次测量，包含启动和写盘）。

| 层 | CSF(G) 记录 | 展开 CSF 数 |
| --- | ---: | ---: |
| AS0 | 2,142 | 2,142 |
| AS1 | 836,618 | 836,618 |
| AS2 | 1,125,587 | 2,990,800 |
| AS3 | 1,125,587 | 6,478,507 |
| AS4 | 1,125,587 | 11,299,739 |

用此前从原版 `e157d9f` 构建的 `rcsfgsplit_csfg` 拆分本次 AS4；
AS1–AS4 共八个 `.g/.l` 文件均逐字节相同。
AS4 `.g` SHA-256 仍是此前原版生成参考的
`4ea7b710ad4b8bedcd5740ccc9bd513a63f225064f0e9bbfa3ea9f4280807187`。
AS0 为独立零激发模型，不从 SD 模型拆分。

本地输入、输出、运行脚本和日志保存在忽略目录
`temp/gd-csfg-batch-2026-10-30/`：`batch.toml`、`generation.json`、
`comparison.json`、`split.in`、`split.log`、`validate.py`。
重跑需改用新的工作目录，避免覆盖已有结果。

实现使用各层直接生成，不是从 `.g` 文件读取并重新映射的 Rust 拆分器。
本轮未改动 Rust 算法；全量 Python 测试 396 项及 21 个 subtest 通过。
