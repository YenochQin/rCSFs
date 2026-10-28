# rCSFs Python 3.15 升级与本机验证

日期：2026-10-28。基于提交 `2e2a20cc4fa5e19467d2a99dd6d0d40e92e81a1c` 的工作区修改。
验证平台为 Linux x86_64，普通 CPython 3.15.2（有 GIL），Rust 1.99.0。

## 结果

本机升级验证通过。`rCSFs/.venv` 使用 Python 3.15.2，已安装本次构建的
`rcsfs` wheel；Tools 的 Python 3.14.8 环境及其已安装包保持独立。
最终解释器保存在 uv 的常规管理目录，`.venv` 不依赖 `temp/` 中的临时解释器。

| 检查 | 结果 |
| --- | --- |
| Rust：`cargo test --release --locked` | 176 项通过，0 失败，包含文档测试 |
| Python：`python -m pytest tests -q` | 356 项通过，另有 21 个子测试通过 |
| Python 类型检查：`basedpyright rcsfs/` | 0 errors，0 warnings |
| 从源码包构建普通 cp315 wheel | 成功 |
| `scripts/check_release.py ... --smoke` | 元数据、归档、导入、Parquet、V2 描述符和 CLI 均通过 |
| `twine check --strict` | wheel 与源码包均通过 |
| Cargo 依赖许可与锁文件指纹 | 通过 |
| 完整 `uv sync --frozen --no-editable --group lint` | 通过；安装后的扩展与已测试扩展 SHA-256 一致 |
| `uv pip check` | 全部已安装依赖兼容 |

Python 测试使用从最终 wheel 复制回源码树的扩展；独立导入检查另行确认了
`.venv` 中安装的 Python 包和原生扩展，二者版本均为 1.3.2。
交互式生成测试保留原有预期哈希，并验证 1、2、4 个线程的输出。

## 环境和依赖调整

- [pyproject.toml](../pyproject.toml)：Python 下限、classifier 和类型检查目标为 3.15；类型检查明确使用本仓库 `.venv`。
- Python 运行依赖：Polars 下限提高到 2.1.0，本次实测版本为 2.1.0。
- 测试依赖：补充 PyArrow 26.0.0 下限，本次实测版本为 26.0.0。
- [Cargo.toml](../Cargo.toml) / [Cargo.lock](../Cargo.lock)：PyO3 0.30.1、pyo3-arrow 0.20.0、Arrow/Parquet 59.3.2、Rust NumPy 0.30.0。
- 发布工作流、包检查脚本和 ABI 测试同步到普通 `cp315-cp315`，拒绝旧版和自由线程 ABI。
- 开发说明及 [AGENTS.md](../AGENTS.md) 改为本仓库独立环境；未修改父仓库或 Tools 的环境约束。

Python 版本迁移本身没有修改 `src/` 下的算法或 Python API 实现；后续 CLI
检查项修复另述于文末。
`uv.lock`、`.venv`、构建产物和运行日志仍沿用仓库的忽略规则。

## 测试中发现并处理的问题

### Polars 1.44.2 的 Python 3.15 方法分派

首轮 Python 回归发现 `Series.list.len()` 返回 `None`。以下最小输入不依赖
rCSFs，也能复现：

```python
import polars as pl

series = pl.Series("x", [[1, 2], [3]], dtype=pl.List(pl.Int32))
result = series.list.len()
assert isinstance(result, pl.Series)
assert result.to_list() == [2, 1]
```

同一套 Polars 1.44.2 文件在 Python 3.14.8 下通过，在 Python 3.15.2 下失败。
Polars 与其原生 runtime 版本一致；直接使用 Expr 的列表长度表达式正常。
定位到该版本 `polars.series.utils._is_empty_method` 对 `co_consts` 的假设：
Python 3.15 的空方法使用 `LOAD_COMMON_CONSTANT None`，常量表不再含它所检查的
第二个 `None`，因此没有安装表达式分派包装器，空方法直接返回 `None`。

检查时最新 1.x 为 1.44.2。将依赖更新到 Polars 2.1.0 后，最小复现、15 项
`read_csfs` 测试及最终全套 Python 测试均通过。未修改或削弱原有断言。

### 独立环境缺少原先由 Tools 提供的测试依赖

`tests/generation_parquet_test.py` 使用 `pyarrow.parquet`，但原来的开发依赖没有
声明它。补充 PyArrow 26 后，测试收集正常。两项 CLI 子进程测试还要求包已经
安装到环境中；安装最终 wheel 后，从临时工作目录启动的 CLI 测试通过。

## 可重复操作与本地产物

从本仓库根目录开始，按 [development.md](development.md) 同步本地环境、构建
并刷新 in-tree 扩展，再运行测试：

```sh
uv sync --python 3.15 --no-editable --group lint
source .venv/bin/activate
cargo test --release --locked
python -m pytest tests -q
basedpyright rcsfs/
```

源码包与 wheel 的构建、审计命令见 [releasing.md](releasing.md)。本次保留的
最终输出位于忽略目录 `temp/python315-final-dist/`：

- `rcsfs-1.3.2-cp315-cp315-manylinux_2_34_x86_64.whl`
- `rcsfs-1.3.2.tar.gz`

有效结果日志：`temp/python315-cargo-test.log`、`temp/python315-pytest-final.log`、
`temp/python315-final-wheel-build.log`、`temp/python315-artifact-check.log`、
`temp/python315-twine-check.log`、`temp/python315-typecheck.log`、
`temp/python315-final-sync.log`。
早期测试日志记录了环境尚未补齐时的失败，不代表最终结果。

## 范围和剩余事项

- 仅实际验证了 Linux x86_64 / CPython 3.15.2；Windows、macOS、自由线程构建和其他 Python 版本未运行。
- 本机 wheel 为 `manylinux_2_34`，不能替代发布工作流中 `manylinux_2_28` 基线的验证。三平台 CI 配置已更新，但本次未触发远程工作流。
- 升级验证时曾记录 9 条既有 Ruff 问题，已在下述后续修复中清理。
- 当前仍为未发布的测试版本 1.3.2。正式发布前需选择新版本，保持版本字段一致，再完成三平台构建验证。
- 本次是库兼容与回归验证，没有运行 GRASP 原子结构计算，也不构成物理精度验证。

## 后续 Ruff 修复（2026-10-28）

9 条代码检查项已修复：错误容器类型改用 `TypeError`，并在 CLI 单命令与批处理
入口保留退出码 2 的参数错误行为；日志改用 `datetime.UTC`；批处理回调通过
`functools.partial` 绑定参数；整理导入和测试上下文管理器。此次涉及 Python
源码，Rust 算法与绑定未修改。

新增 7 个非法 TOML 类型用例。同步本地安装后，完整 Python 回归为 **363 项通过、
21 个子测试通过**；类型检查及相关文件的 Ruff 检查通过。最终日志为
`temp/lint-fixes-pytest-full.log` 和 `temp/lint-fixes-sync.log`。
Ruff 0.16.7 对 Python 3.15 的支持状态提示仍可能输出，它独立于已清零的代码诊断。

随后对 `tests/` 和 `scripts/` 完整扫描，另清理 40 条 Ruff 检查项，包括导入、
无效 `noqa`、异常类型、条件合并、正则标志与子进程参数写法。补充基准清单和
描述符比较工具的 7 个输入校验测试后，完整 Python 回归为 **370 项通过、
21 个子测试通过**，`ruff check rcsfs/ tests/ scripts/` 全部通过。
最终回归日志为 `temp/ruff-tests-scripts-pytest.log`。没有运行外部 Fortran 基准计算。
