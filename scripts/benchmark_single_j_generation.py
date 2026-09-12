"""Reproduce the local single-J benchmark; requires built release examples and GRASP.

Run from rCSFs with uv run python scripts/benchmark_single_j_generation.py.
Writes the dated benchmark report and raw measurements under docs/benchmarks.
"""

from pathlib import Path
import subprocess
import tempfile
import os
import hashlib
import json
import statistics
import platform

root = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


rows = []
with tempfile.TemporaryDirectory(prefix="rcsfs-single-j-") as tmp:
    work = Path(tmp)
    transcript = (
        (root / "tests/fixtures/e1_cc1as1.rcsfgenerate")
        .read_text()
        .replace("0,12", "6,6")
    )
    inp = work / "single_j.rcsfgenerate"
    inp.write_text(transcript)
    original = work / "fortran"
    original.mkdir()
    subprocess.run(
        [str(root.parent / "grasp/build-debug/bin/rcsfgenerate")],
        input="\n".join(transcript.splitlines()[1:-1]) + "\n",
        text=True,
        cwd=original,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=True,
        timeout=120,
    )
    reference_hash = digest(original / "rcsf.out")
    for threads in [1, 2, 4]:
        for run in range(4):
            out = work / f"{threads}-{run}.c"
            result = subprocess.run(
                [
                    str(root / "target/release/examples/benchmark_generation"),
                    str(inp),
                    str(out),
                ],
                env={**os.environ, "RCSFS_THREADS": str(threads)},
                capture_output=True,
                text=True,
                check=True,
                timeout=120,
            )
            values = {}
            for line in result.stdout.splitlines():
                if " = " in line:
                    key, value = line.split(" = ", 1)
                    try:
                        values[key] = float(value)
                    except ValueError:
                        pass
            assert digest(out) == reference_hash
            rows.append(dict(threads=threads, warmup=run == 0, **values))
    data = {
        "platform": platform.platform(),
        "input": "e1_cc1as1.rcsfgenerate with 2J range changed from 0,12 to 6,6",
        "input_sha256": hashlib.sha256(transcript.encode()).hexdigest(),
        "output_sha256": reference_hash,
        "runs": rows,
    }
    target = root / "docs/benchmarks/rcsfgenerate_parallel_20260912.json"
    target.write_text(json.dumps(data, indent=2) + "\n")
    medians = {
        n: {
            k: statistics.median(
                r[k] for r in rows if r["threads"] == n and not r["warmup"]
            )
            for k in ("generation_seconds", "total_seconds")
        }
        for n in [1, 2, 4]
    }
    report = """# 本机单 J 并行回归（2026-09-12）

输入由登记的 e1_cc1as1 transcript 将 `0,12` 改为 `6,6`，其他回答不变。
原版使用 workspace 中固定版本的 `grasp/build-debug/bin/rcsfgenerate` 重新生成；Rust 每份输出均与它逐字节 SHA-256 相同。此处原版只作正确性对照，不比较 Debug Fortran 与 Release Rust 的速度。

硬件：Apple M4，10 核，16 GiB；Rust `cargo build --release --examples`，默认 release 优化。每线程数预热 1 次，随后测量 3 次，表中为中位数。顺序测量 1/2/4 线程，缓存未清空；未测物理 I/O、服务器 RSS 或高线程数。这是本机单 J 验证，不能推广为服务器性能结论。

|线程|生成阶段（秒）|端到端（秒）|生成加速|端到端加速|
|---|---|---|---|---|
"""
    for n, v in medians.items():
        report += f"|{n}|{v['generation_seconds']:.6f}|{v['total_seconds']:.6f}|{medians[1]['generation_seconds'] / v['generation_seconds']:.2f}×|{medians[1]['total_seconds'] / v['total_seconds']:.2f}×|\n"
    report += f"\n记录数：{int(rows[0]['records'])}；唯一占据任务：{int(rows[0]['unique_occupations'])}。\n\n输出 SHA-256：`{reference_hash}`。\n\n[原始测量](rcsfgenerate_parallel_20260912.json)。生成耗时包括结果组织所需的分支归并；端到端包括输入、占据枚举、生成和最终文本写出。前缀细分只覆盖至少 64 个态组合的任务，不代表所有单占据任务都能充分并行。\n"
    (root / "docs/benchmarks/rcsfgenerate_parallel_20260912.md").write_text(report)
    print(report)
