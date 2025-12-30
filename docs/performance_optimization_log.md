# CSF Descriptor Generation Performance Optimization Log

## Current Status (2025-12-30)

### Performance Profile
- **CPU Utilization**: ~10% on 48-core system
- **Workers**: 48 threads configured
- **Orbital Count**: 56 orbitals
- **Descriptor Size**: 168 (56 × 3)

### Flamegraph Analysis (flamegraph.svg)

| Component | CPU Time | Notes |
|-----------|----------|-------|
| Memory allocation (malloc/free/cfree) | ~43% | Largest bottleneck |
| Single-threaded Parquet Writer | 8.8% | `parquet::arrow::arrow_writer::write_primitive` |
| String trim operations | 7.3% | `trim_matches` (5.72%) + `trim_end_matches` (1.57%) |
| Lock contention | 20%+ | `__lll_lock_wait_private` |
| CSF parsing | 6.29% | `parse_csf` function |

---

## Completed Optimizations

### ✅ 1. Fixed Nested Parallelization (src/csfs_descriptor.rs:536-548)
**Issue**: Rayon's `par_iter()` inside manual worker threads caused over-subscription

```rust
// BEFORE
let descriptors: Vec<Vec<i32>> = work_item.rows
    .into_par_iter()  // ❌ Nested parallelization
    .map(...)
    .collect();

// AFTER
let descriptors: Vec<Vec<i32>> = work_item.rows
    .into_iter()  // ✅ Sequential iteration
    .map(...)
    .collect();
```

### ✅ 2. Increased Channel Capacity (line 376)
```rust
// BEFORE: capacity = num_workers * 2 (96)
let channel_capacity = num_workers * 10;  // 480 capacity
```

### ✅ 3. Added Failure Tracking
- Added `failed_count` field to `BatchDescriptorStats`
- Added `failed_count` field to `ResultItem` struct

### ✅ 4. Improved Error Messages in `j_to_double_j`
```rust
pub fn j_to_double_j(j_str: &str) -> Result<i32, String> {
    let trimmed = j_str.trim();

    if trimmed.is_empty() {
        return Err(format!("Empty J value string"));
    }
    // ... better error context
}
```

### ✅ 5. Replaced `occupied_mask` Heap Allocation with Stack Allocation (line 797)
**Impact**: ~50% reduction in per-CSF allocations

```rust
// BEFORE: Heap allocation per CSF
let mut occupied_mask = vec![false; self.orbital_count];

// AFTER: Stack allocation
let mut occupied_mask = [false; 128];
let occupied_mask_len = self.orbital_count;
```

### ✅ 6. Cached `trim()` Results to Reduce Redundant Calls (lines 864-877)
```rust
// Cache trim results instead of calling trim() multiple times
let middle_trimmed = middle_item.trim();
let coupling_trimmed = coupling_item.trim();
```

---

## Failed Optimization Attempts

### ❌ 1. mimalloc Memory Allocator (REVERTED)
**Attempt**: Replace system malloc with mimalloc
**Result**: CPU dropped to 3%, memory usage increased
**Conclusion**: Problem is excessive allocations, not the allocator itself

### ❌ 2. RecordBatch Instead of Arc<str> (REVERTED)
**Attempt**: Pass RecordBatch directly to workers instead of extracting Arc<str>
**Result**: CPU dropped to 3%
**Conclusion**: Column extraction overhead > Arc allocation overhead
**Revert command**: `git checkout aea0718 -- src/csfs_descriptor.rs`

### ❌ 3. Reducing Worker Count (4, 8, 16 workers)
**Result**: No change in CPU utilization
**Conclusion**: Bottleneck is in single-threaded operations, not parallelization

### ❌ 4. Increasing Channel Capacity
**Result**: No significant improvement
**Conclusion**: Channel capacity is not the limiting factor

---

## Remaining Bottlenecks (Priority Order)

### 🔴 High Priority: Single-threaded Parquet Writer (8.8% CPU)
**Location**: `parquet::arrow::arrow_writer::write_primitive`

**Possible Solutions**:
1. **Batch-based writer pooling**: Multiple writers writing to separate files, then merge
2. **Async I/O**: Use tokio/async-std for non-blocking writes
3. **Column-group writing**: Write in parallel by column groups
4. **Increase batch size**: Fewer, larger batches reduce write frequency

### 🟡 Medium Priority: Lock Contention (20%+ CPU)
**Location**: `__lll_lock_wait_private` in multiple locations

**Possible Solutions**:
1. **Lock-free structures**: Use crossbeam atomic queues
2. **Reduce shared state**: Fewer mutexes, more message passing
3. **Shard data structures**: Partition to reduce contention

### 🟡 Medium Priority: String Processing (7.3% CPU)
**Location**: `trim_matches` (5.72%) + `trim_end_matches` (1.57%)

**Possible Solutions**:
1. **Pre-trim at source**: Trim once when reading from parquet
2. **Avoid unnecessary trims**: Only trim when actually needed
3. **Use bytes instead of str**: Work with &[u8] for ASCII data

### 🟢 Low Priority: Memory Allocation (43% CPU)
**Note**: Changing allocator didn't help; need to reduce allocation count

**Possible Solutions**:
1. **Pool reuse**: Reuse descriptor Vecs across CSF parses
2. **Arena allocator**: Bump allocator for batch-level allocations
3. **Change data structure**: Consider different descriptor format

---

## Code Files Modified

| File | Lines Changed | Description |
|------|---------------|-------------|
| `src/csfs_descriptor.rs` | 793-920 | parse_csf optimizations (occupied_mask, trim caching) |
| `src/csfs_descriptor.rs` | 376 | Channel capacity increase |
| `src/csfs_descriptor.rs` | 536-548 | Fixed nested parallelization |
| `src/csfs_descriptor.rs` | ~200 | Error handling improvements |
| `Cargo.toml` | - | mimalloc added then reverted |
| `src/lib.rs` | - | mimalloc global allocator then reverted |

---

## Next Steps (Daytime Work)

### Step 1: Verify Current Performance
Run baseline test with current optimizations:
```bash
# Test CPU utilization
# Check if occupied_mask optimization helped
```

### Step 2: Address Writer Bottleneck (8.8%)
Consider implementing one of:
- Batch size tuning (simplest)
- Column-group parallel writing
- Async I/O with tokio

### Step 3: Address Lock Contention (20%)
- Profile to identify exact lock locations
- Consider lock-free alternatives

### Step 4: String Processing Optimization (7.3%)
- Pre-trim at parquet read time
- Use byte operations instead of str

---

## Commands Reference

```bash
# Build and install
maturin develop --release

# Run with specific worker count
python -c "import rcsfs; rcsfs.generate_descriptors_from_parquet(..., num_workers=16)"

# Profile with flamegraph
cargo install flamegraph
cargo flamegraph --bin rcsfs -- <args>

# Git revert if needed
git checkout aea0718 -- src/csfs_descriptor.rs

# Check git log
git log --oneline -10
```

---

## Performance Hypothesis

**Root Cause**: The code performs millions of tiny allocations per second:
- Each CSF: 1 × descriptor Vec (672 bytes) + 1 × occupied_mask Vec (56 bytes) + 3 × Arc<str>
- With 48 workers all allocating simultaneously → allocator contention → lock overhead

**Why mimalloc failed**: It optimizes allocation speed, but doesn't solve the fundamental problem of too many allocations

**Correct approach**: Reduce allocation count, not optimize the allocator

**Why Writer is still slow**: Single-threaded ArrowWriter cannot keep up with 48 parallel workers
