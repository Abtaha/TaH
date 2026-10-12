"""CPU-only Triton interpretation/offline compilation; never loads a GPU module."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['interpret', 'compile'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        parser.error('CPU tooling requires CUDA_VISIBLE_DEVICES=""')
    if args.mode == 'interpret':
        os.environ['TRITON_INTERPRET'] = '1'
    import torch
    import triton
    from tah.engine.attention_kernels import count_visible, pack_visible
    report = dict(production_gates_eligible=False, mode=args.mode, cases=[])
    if args.mode == 'compile':
        from triton.compiler import ASTSource
        from triton.backends.compiler import GPUTarget
        for arch in [80, 86, 89]:
            for fn, names in [(count_visible, ['pos', 'depths', 'starts', 'lengths', 'query_pos', 'query_depth', 'counts']),
                (pack_visible, ['slots', 'pos', 'depths', 'starts', 'lengths', 'query_pos', 'query_depth', 'indptr', 'indices'])]:
                signature = {name: '*i64' if name in ['pos', 'query_pos'] else '*i32' for name in names}
                compiled = triton.compile(ASTSource(fn, signature, constants={'BLOCK': 256}), target=GPUTarget('cuda', arch, 32))
                report['cases'].append(dict(arch=arch, kernel=fn.__name__, ptx_bytes=len(compiled.asm['ptx']),
                                            cubin_bytes=len(compiled.asm['cubin'])))
    else:
        for seed in range(20):
            gen = torch.Generator().manual_seed(seed)
            lengths = torch.tensor([1, 7, 255, 256, 257, 513, 2049], dtype=torch.int32)
            starts = torch.cat([torch.zeros(1, dtype=torch.int32), lengths.cumsum(0).int()[:-1]])
            total = int(lengths.sum())
            pos = torch.randint(2000, (total,), generator=gen)
            depths = torch.randint(1, 4, (total,), generator=gen, dtype=torch.int32)
            slots = torch.randperm(total, generator=gen).int()
            qp = torch.randint(2000, (len(lengths),), generator=gen)
            qd = torch.randint(1, 4, (len(lengths),), generator=gen, dtype=torch.int32)
            counts = torch.empty_like(lengths)
            count_visible[(len(lengths),)](pos, depths, starts, lengths, qp, qd, counts, BLOCK=256)
            ptr = torch.cat([torch.zeros(1, dtype=torch.int32), counts.cumsum(0).int()])
            ids = torch.full((total,), -1, dtype=torch.int32)
            pack_visible[(len(lengths),)](slots, pos, depths, starts, lengths, qp, qd, ptr, ids, BLOCK=256)
            for row in range(len(lengths)):
                sl = slice(int(starts[row]), int(starts[row] + lengths[row]))
                expected = slots[sl][(pos[sl] <= qp[row]) & (depths[sl] <= qd[row])]
                assert int(counts[row]) == len(expected)
                assert torch.equal(ids[int(ptr[row]):int(ptr[row + 1])], expected)
            assert bool((ids[int(ptr[-1]):] == -1).all()), 'compaction wrote outside packed result'
            report['cases'].append(dict(seed=seed, rows=len(lengths), sources=total, visible=int(ptr[-1])))
    assert not torch.cuda.is_initialized()
    report['cuda_initialized'] = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f'{args.mode}: {len(report["cases"])} cases PASS; CUDA uninitialized')


if __name__ == '__main__':
    main()
