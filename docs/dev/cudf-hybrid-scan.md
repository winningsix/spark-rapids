# cuDF Hybrid Scan for Parquet

## Overview

cuDF Hybrid Scan is an advanced Parquet reader optimization that uses a two-step (late) materialization process for highly-selective filter expressions. Instead of reading all columns at once, it:

1. **First step**: Materialize only filter columns, evaluate filters, build a survival row mask
2. **Second step**: Use the survival mask to skip pages of payload columns, then materialize payload
3. **Combine**: Merge filter and payload columns into the final result

## Configuration

| Configuration | Default | Description |
|---------------|---------|-------------|
| `spark.rapids.sql.parquet.cudfHybridScan.enabled` | `false` | Enable cuDF Hybrid Scan for Parquet reads |
| `spark.rapids.sql.parquet.cudfHybridScan.pipelining.enabled` | `true` | Enable intra-file I/O pipelining |
| `spark.rapids.sql.parquet.cudfHybridScan.parallel.enabled` | `true` | Enable parallel row group I/O |
| `spark.rapids.sql.parquet.cudfHybridScan.parallel.maxThreads` | `4` | Max threads for parallel row group I/O |

## When Hybrid Scan is Beneficial

Hybrid Scan provides the most benefit when:

1. **Low selectivity queries** (< 5% of rows pass the filter)
   - Two-step materialization saves I/O by reading filter columns first
   - Example: `WHERE status = 'CANCELLED'` when <5% of orders are cancelled

2. **Page-level indexes available** (Parquet 2.0+)
   - Allows skipping entire pages of payload columns
   - Requires data written with page statistics enabled

3. **Wide tables with few filter columns**
   - Larger benefit when payload columns >> filter columns
   - Example: 100 columns with filter on 2 columns

## When Hybrid Scan May NOT Help

1. **High selectivity queries** (> 10% of rows pass)
   - Two-step materialization overhead exceeds savings
   - TPC-H Q1 (98% selectivity) and Q6 (~10% selectivity) show this

2. **Parquet Format 1.0 without page indexes**
   - Cannot skip pages within row groups
   - Only row-group level statistics available

3. **All columns used in filter**
   - No payload columns to skip

## Implementation Details

### GPU OOM Retry Support

The Hybrid Scan reader is integrated with RAPIDS OOM framework:
- Uses `RmmRapidsRetryIterator.withRetryNoSplit` for automatic retry on GPU OOM
- Supports spilling and rollback during memory pressure

### Semaphore Optimization

To maximize GPU concurrency:
- Releases GPU semaphore during I/O operations
- Re-acquires before GPU materialization
- Allows other tasks to use GPU during I/O wait

### Parallel I/O for Filter and Payload Columns

Both filter and payload column I/O are submitted concurrently:
- Uses thread pool for parallel I/O submission
- Registers I/O threads with RMM for OOM handling
- Overlaps filter column I/O with payload column I/O

## Performance Test Results

### Test Environment
- TPC-H SF100 (600M rows in lineitem)
- Parquet Format 1.0 (no page index)
- Single node, 8 cores, L20 GPU

### Results

| Query | Hybrid OFF | Hybrid ON | Speedup |
|-------|------------|-----------|---------|
| Q1 (98% selectivity) | 1.089s | 3.930s | 0.28x (slower) |
| Q6 (~10% selectivity) | 0.688s | 0.783s | 0.88x (slower) |

### Analysis

For TPC-H queries without page index:
- **Q1**: Very high selectivity (98% rows pass filter), two-step overhead dominates
- **Q6**: Medium selectivity (~10%), overhead still exceeds benefit

**Conclusion**: Hybrid Scan benefits require either very low selectivity (<5%) or page-level indexes for page skipping. For typical TPC-H queries on Format 1.0 data, the feature should remain disabled.

## Future Work

1. **Automatic selectivity estimation**: Use row group statistics to estimate selectivity and auto-enable
2. **Page index detection**: Automatically detect and use page-level indexes when available
3. **Adaptive threshold**: Dynamically adjust selectivity threshold based on data characteristics
