# CSF 与 CSFG 生成源码布局

2026-10-30：两套生成流程复用同一占据枚举、子壳层状态表和角动量耦合算法，
因此统一归入 `src/configuration_generation/`，其专用流程使用平级子目录。

```text
src/configuration_generation/
├── mod.rs
├── shared_generation/
│   ├── mod.rs
│   ├── angular_coupling_generator.rs
│   ├── electron_occupation_enumeration.rs
│   ├── subshell_state_catalog.rs
│   ├── csf_record_counting.rs
│   ├── generation_task_planning.rs
│   └── generation_execution_options.rs
├── csf_generation/
│   ├── mod.rs
│   ├── complete_csf_generation_pipeline.rs
│   ├── disk_csf_generation_pipeline.rs
│   ├── csf_artifact_encoding.rs
│   ├── csf_output_capacity_estimation.rs
│   ├── disk_space_preflight.rs
│   └── python_csf_bindings.rs
└── csfg_generation/
    ├── mod.rs
    ├── direct_csfg_generation_pipeline.rs
    ├── representative_occupation_planning.rs
    ├── graspg_csfg_output_writer.rs
    ├── python_csfg_bindings.rs
    └── csfg_compatibility_tests.rs
```

## 职责与依赖

- **共享生成模块**：相对论占据配置、状态表与耦合递归。通用记录计数、任务规划和资源设置也放在这里，
  其中精确计数/任务规划目前主要供完整 CSF 的磁盘路径使用。
- **完整 CSF 模块**：内存/磁盘完整列表生成、V2/CSF 产物编码、磁盘容量估算及空间预检。
- **CSFG 模块**：五类代表占据规划、直接代表态生成、GRASPG 配套文件编码与相应 Python 绑定。
- 两套流程均调用共享算法；CSFG 没有依赖完整 CSF 的文件写出、V2 或磁盘流水线。
- `src/complete_csf.rs`、`src/csf_output.rs` 等全库数据格式与输出工具仍为转换、恢复、筛选等功能共用，
  不属于某一种生成流程的专属实现。

每个业务源文件名都说明领域和职责。`mod.rs` 是 Rust 约定的模块入口，仅声明/导出；
原来放在大型 `csf_generation/mod.rs` 中的计算实现已移入 `angular_coupling_generator.rs`。
生成相关的 PyO3 函数体已归入两套生成目录中的 `python_*_bindings.rs`；`src/lib.rs` 保留它们的注册。

## 主要名称对应

| 原名称 | 新名称 |
| --- | --- |
| `csf_generation/mod.rs` 中的耦合实现 | `shared_generation/angular_coupling_generator.rs` |
| `occupations.rs` | `shared_generation/electron_occupation_enumeration.rs` |
| `states.rs` | `shared_generation/subshell_state_catalog.rs` |
| `counting.rs` | `shared_generation/csf_record_counting.rs` |
| `planning.rs` | `shared_generation/generation_task_planning.rs` |
| `options.rs` | `shared_generation/generation_execution_options.rs` |
| `pipeline.rs` | `csf_generation/complete_csf_generation_pipeline.rs` |
| `streaming.rs` | `csf_generation/disk_csf_generation_pipeline.rs` |
| `final_encoding.rs` | `csf_generation/csf_artifact_encoding.rs` |
| `capacity.rs` | `csf_generation/csf_output_capacity_estimation.rs` |
| `space.rs` | `csf_generation/disk_space_preflight.rs` |
| `csfg/mod.rs` 中的流程实现 | `csfg_generation/direct_csfg_generation_pipeline.rs` |
| `csfg/plan.rs` | `csfg_generation/representative_occupation_planning.rs` |
| `csfg/writer.rs` | `csfg_generation/graspg_csfg_output_writer.rs` |
| `csfg/tests.rs` | `csfg_generation/csfg_compatibility_tests.rs` |
| `csfg_py.rs` | `csfg_generation/python_csfg_bindings.rs` |
| `lib.rs` 中的完整 CSF 生成绑定 | `csf_generation/python_csf_bindings.rs` |

## 接口与验证

新的 Rust 分组入口为 `configuration_generation`。既有 `csf_generation` 是它的兼容导出，
既有 `csf_generation::csfg` 指向新 CSFG 子模块，不包含重复实现。Python 函数、CLI 命令和 TOML
仍使用既有名称；这次修改只涉及源码组织、模块引用与绑定归属。

验证使用已有 Rust/Python 测试、原版小模型 fixture 和同一 Gd 输入的输出哈希。
目录整理不应改变任何 CSF/CSFG 顺序、计数或输出字节；性能与物理验证的边界仍见
[CSFG 使用说明](CSFG_GENERATION.md)和[Gd 基准](benchmarks/csfg_gd_20261030.md)。

本次验证：76 项 Rust 单元测试、386 项 Python 测试和 21 个子测试通过；
`cargo check --all-targets`、Ruff、basedpyright 和 Rust 格式检查通过。
同一 Gd 输入重新生成的完整 CSF、CSFG 与标签文件，均与整理前基准逐字节一致。
占据枚举、状态表、执行选项及计数文件原样移动；耦合内核仅调整两处内部可见性以供同级模块使用。
验证记录保存在忽略的 `temp/generation-layout-before-2026-10-30/validation.json`。
