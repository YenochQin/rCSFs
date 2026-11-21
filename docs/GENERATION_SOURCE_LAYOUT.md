# CSF 生成源码布局

Rust 生成入口位于 [configuration_generation](../src/configuration_generation/mod.rs)。
`csf_generation` 是这个模块的兼容名称。

| 生成阶段 | 代码入口 |
| --- | --- |
| 根据参考组态与激发规则枚举占据 | [electron_occupation_enumeration.rs](../src/configuration_generation/shared_generation/electron_occupation_enumeration.rs) |
| 查找子壳层态并枚举角动量耦合 | [subshell_state_catalog.rs](../src/configuration_generation/shared_generation/subshell_state_catalog.rs) 和 [angular_coupling_generator.rs](../src/configuration_generation/shared_generation/angular_coupling_generator.rs) |
| 在内存生成完整 CSF 并写出描述符 | [complete_csf_generation_pipeline.rs](../src/configuration_generation/csf_generation/complete_csf_generation_pipeline.rs) |
| 分批生成、归并磁盘记录并写出描述符 | [disk_csf_generation_pipeline.rs](../src/configuration_generation/csf_generation/disk_csf_generation_pipeline.rs) |
| 估算磁盘容量并检查可用空间 | [csf_output_capacity_estimation.rs](../src/configuration_generation/csf_generation/csf_output_capacity_estimation.rs) 和 [disk_space_preflight.rs](../src/configuration_generation/csf_generation/disk_space_preflight.rs) |

两条生成路径复用同一套占据枚举、子壳层态表和角动量耦合算法。输入格式与调用示例见 [CSF 生成指南](CSF_GENERATION.md)，V3 行的字段定义见[描述符指南](CSF_DESCRIPTOR_GUIDE.md)。
