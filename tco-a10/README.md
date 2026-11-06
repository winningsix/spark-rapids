# TPC-DS SF1000 on A10: 2-GPU vs 4-GPU performance and TCO

This branch (`A10_TCO`) holds the cudf-spark source changes behind the frozen
4xA10 package, plus the harness, the driver overlay, the analysis scripts and the
measured results used to compare 2-GPU and 4-GPU A10 configurations against a
Gluten (Velox) CPU baseline.

- Base: `v26.08.1-rc` (1253310). Spark 4.0.2, Scala 2.13, JDK 17.
- Plugin source changes: the commit "TPC-DS SF1000 A10 optimizations" (36 files;
  they match the frozen package's `source-manifest.json`, plugin JAR sha256 `680faaf5...`).

## Protocol

- TPC-DS SF1000 Parquet, 103 queries, original SQL stream.
- Each query: 1 warmup + 3 measured. The timer stops at `collect()`.
  Metric = **sum of per-query medians**.
- Answers compared row by row against the 4-GPU reference (Q39a/b within 1e-12).
- Hardware: AWS g5 (4x A10G 24 GB). AWS has no 2-GPU g5 size, so 2 GPUs are
  emulated on the same host with cgroup v2: cpuset 0-27,48-75 (56 vCPU), 224 GiB
  memory, page cache dropped, `spark.cores.max=20`, `spark.executor.instances=2`.
  Each executor keeps 10 cores, 1 GPU and the same memory settings.

## Results (absolute times)

| Configuration | 4 GPU | 2 GPU | 2 GPU / 4 GPU |
|---|---|---|---|
| Frozen package JAR (this branch's source) | 202.93 s | 272.63 s | 1.34x |
| Later best JAR (frequency-aggregate-v2 overlay chain) | 191.47 s | 251.98 s | 1.32x |
| Later best JAR + closure-check overlay | **184.33 s** | **244.93 s** | 1.33x |

All runs: 412/412 records OK, answers equal to the reference, no task failures,
retries, spill, executor loss or OOM.

The closure-check overlay passed a predeclared ABBA gate on 4 GPUs (1.039x and 1.040x).
It saves about 7 s of driver time between stages on both 2 and 4 GPUs.

## GPU utilization (DCGM, only the GPUs in use, during measured queries)

| Metric | 4 GPU (+overlay) | 2 GPU (+overlay) | 2 GPU frozen JAR |
|---|---|---|---|
| GPU util | 29% | 46% | ~55% |
| SM active | 0.16 | 0.27 | 0.38 |
| SM occupancy | 0.10 | 0.18 | ~0.24 |
| DRAM active | 0.10 | 0.17 | ~0.23 |
| Samples fully idle | 51% | 37% | ~19% |

On 4 GPUs the GPUs mostly wait on the driver and CPU-side work. Driver time with
no task running is about 30-37 s per run regardless of GPU count. That fixed cost
is why halving the GPUs costs only about 1.33x in time.

## TCO versus Gluten

- Gluten baseline: Tencent EMR, Spark 3.5.3 + Gluten 1.5.0-SNAPSHOT (Velox), 31 executors x 4 cores.
  From its event log, the sum of per-query medians over 3 rounds is 561.68 s, at 6.9 RMB per run.
- GPU price: Tencent instance list price.
  - 4xA10 (PNV4.28XLARGE466): 42.82-47.57 RMB/h
  - 2xA10 (PNV4.14XLARGE232): 21.38-23.75 RMB/h
- Cost per run = price x sum of medians / 3600.

| GPU configuration | Time | Speedup vs Gluten | RMB/run | TCO vs Gluten |
|---|---|---|---|---|
| 4 GPU frozen JAR | 202.93 s | 2.77x | 2.41-2.68 | 2.57-2.86x |
| 4 GPU + overlay | 184.33 s | 3.05x | 2.19-2.44 | 2.83-3.15x |
| 2 GPU frozen JAR | 272.63 s | 2.06x | 1.62-1.80 | 3.84-4.26x |
| 2 GPU + overlay | 244.93 s | 2.29x | 1.45-1.62 | 4.27-4.74x |

Caveat: in 99 of the 103 queries, the Gluten stream used different query literals.

## Layout

- `harness/scripts/`: `benchctl.py` (`prepare --gpus 2|4`), `run_benchmark.py`, `validate_results.py`.
- `harness/config/`: `spark-202s.properties.in` (4-GPU template), `spark-2gpu-example.properties`.
- `closure-check-overlay/`: driver-only `SparkClosureCleaner$` replacement and `build.sh`.
- `analysis/`: `analyze-gap-split-v1.py` (driver gaps from event logs),
  `dcgm-logger-v1.sh` + `analyze-dcgm-v1.py` (utilization aligned to query windows).
- `results/`: review, validation, DCGM and gap-split outputs of the runs above.

The "later best JAR" rows come from a chain of class-level overlays built on top
of this source. Those overlays are not yet part of this branch's source.
