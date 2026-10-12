"""CPU/CUDA ragged attention and swap-buffer microbenchmarks, JSON output."""
import argparse
import json
from pathlib import Path
import time

import torch

from tah.engine.attention import DecodeBucket, PrefillAttention, AttentionRow
from tah.engine.kv_cache import SlotPool


def measure(fn, device, repeats):
    for _ in range(3):
        fn()
    if device == 'cuda':
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) * 1000 / repeats
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - start) * 1e6 / repeats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    torch.set_num_threads(2)
    results = []
    for batch in [1, 8, 32]:
        if args.device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        lengths = [31 + 17 * i for i in range(batch)]
        pool = SlotPool(sum(lengths) * 3, 1, 8, 128, 3, device=args.device)
        rows = []
        for i, length in enumerate(lengths):
            request = pool.create_request(i, context_limit=length + 1)
            for d in range(3):
                pool.allocate(request, d, list(range(d, length, d + 1)))
            rows.append(AttentionRow(request, length - 1, 1 + i % 3))
        pool.values.normal_()
        q = torch.randn(batch, 16, 128, device=args.device, dtype=torch.bfloat16)
        bucket = DecodeBucket(pool, batch, sum(r.request.size for r in rows), 16)
        bucket.prepare(rows)
        record = dict(batch=batch, real_positions=sum(lengths), kv_entries=sum(r.request.size for r in rows),
            prepare_cpu_wall_us=measure(lambda: bucket.prepare(rows), 'cpu', args.repeats),
            metadata_us=measure(bucket.build, args.device, args.repeats),
            decode_us=measure(lambda: bucket.run(q, 0), args.device, args.repeats))
        chunk_rows = [AttentionRow(r.request, p, r.depth) for r, length in zip(rows, lengths) for p in range(length - 8, length)]
        prefill = PrefillAttention(pool, chunk_rows, 16)
        chunk_q = q.repeat_interleave(8, 0)
        record['prefill_8_us'] = measure(lambda: prefill.run(chunk_q, 0), args.device, args.repeats)
        request = rows[-1].request
        snap = pool.gather(request)
        record['gather_us'] = measure(lambda: pool.gather(request, out=snap.values), args.device, args.repeats)
        record['scatter_us'] = measure(lambda: pool.scatter(request, snap), args.device, args.repeats)
        record['payload_bytes'] = snap.values.numel() * snap.values.element_size()
        if args.device == 'cuda':
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                bucket.build()
                bucket.run(q, 0)
            record['metadata_decode_graph_us'] = measure(graph.replay, args.device, args.repeats)
            record['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        for r in rows:
            pool.release(r.request)
        record['pool_audit'] = pool.audit()
        results.append(record)
        if args.device == 'cuda':
            torch.cuda.synchronize()
            del graph
        del pool, bucket, prefill, q, chunk_q, snap, request, rows, chunk_rows
    report = dict(production_gates_eligible=False, device=args.device, dtype='bfloat16', threads=2,
                  repeats=args.repeats, units='microseconds', cases=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if args.device == 'cpu':
        assert not torch.cuda.is_initialized()


if __name__ == '__main__':
    main()
