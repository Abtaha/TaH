"""Device-only ragged duo-visibility count and stable compaction.

One program per query row loops over its OWN real KV entries, tile by tile.
No batch-by-longest buffers; no dynamic allocation or host reads in kernels.
"""
import triton
import triton.language as tl


@triton.jit
def count_visible(pos, depths, starts, lengths, query_pos, query_depth, counts,
                  BLOCK: tl.constexpr):
    row = tl.program_id(0)
    start = tl.load(starts + row)
    length = tl.load(lengths + row)
    p = tl.load(query_pos + row)
    d = tl.load(query_depth + row)
    lane = tl.arange(0, BLOCK)
    total = tl.full((), 0, tl.int32)
    for base in range(0, length, BLOCK):
        offset = base + lane
        kp = tl.load(pos + start + offset, offset < length, other=2147483647)
        kd = tl.load(depths + start + offset, offset < length, other=2147483647)
        valid = (offset < length) & (kp <= p) & (kd <= d)
        total += tl.sum(valid.to(tl.int32), 0)
    tl.store(counts + row, total)


@triton.jit
def pack_visible(slots, pos, depths, starts, lengths, query_pos, query_depth,
                 indptr, indices, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    start = tl.load(starts + row)
    length = tl.load(lengths + row)
    p = tl.load(query_pos + row)
    d = tl.load(query_depth + row)
    dest = tl.load(indptr + row)
    lane = tl.arange(0, BLOCK)
    for base in range(0, length, BLOCK):
        offset = base + lane
        kp = tl.load(pos + start + offset, offset < length, other=2147483647)
        kd = tl.load(depths + start + offset, offset < length, other=2147483647)
        valid = (offset < length) & (kp <= p) & (kd <= d)
        rank = tl.cumsum(valid.to(tl.int32), 0) - 1
        slot = tl.load(slots + start + offset, offset < length, other=0)
        tl.store(indices + dest + rank, slot, valid)
        dest += tl.sum(valid.to(tl.int32), 0)
