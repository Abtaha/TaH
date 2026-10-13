#!/usr/bin/env python3
"""CPU-only extraction of the sufficient metrics used by the chapter.

Original outputs remain in experiment directories. Source paths/hashes are
recorded per curated file; no result or validation threshold is fitted here.
"""
import argparse
import hashlib
import json
import lzma
from pathlib import Path
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parents[1]
DATA=ROOT/'data'
CHECKPOINTS=('real','mlp3','mlp1','label3')


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')


def select(value,keys):
    return {k:value[k] for k in keys.split() if k in value}


def performance(value):
    result=select(value,'status device workload requests batch_size max_new_tokens peak_gpu_allocated_bytes peak_gpu_reserved_bytes cuda_graphs compile overlap runtime_sha256')
    result['run']=select(value['run'],'generated_tokens wall_seconds tokens_per_second')
    return result


def parity(value):
    return select(value,'checkpoint status summary noise_limits decision_limits generated_positions limits_sha256 limits_saved_unix_time comparison_unix_time')


def profile(value):
    return select(value,'decode_steps context unprofiled_mean_step_ms mean_kernel_gpu_ms mean_gpu_copy_ms estimated_unprofiled_python_launch_idle_ms overlap_host_ms runtime_sha256')


def robustness(value):
    result=select(value,'status requests callbacks max_new_tokens total_prompt_tokens kv_pool_high_water_entries output_tokens decisions_identical kv_audit host_swap_used_bytes runtime_sha256')
    result['run']=select(value['run'],'generated_tokens swap_outs swap_ins restarts wasted_tokens host_swap_peak_bytes')
    if 'results' in value:
        result['max_prompt_tokens']=max(len(r.get('prompt_pass_counts',[])) for r in value['results'])
    return result


class Collector:
    def __init__(self,sources):
        self.sources=sources
        self.used=[]

    def load(self,key):
        path=Path(self.sources[key]);path=path if path.is_absolute() else REPO/path
        b=path.read_bytes();self.used.append(dict(source_path=str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path),source_sha256=hashlib.sha256(b).hexdigest(),source_bytes=len(b),source_mtime_utc=datetime.fromtimestamp(path.stat().st_mtime,timezone.utc).isoformat()))
        return json.loads(lzma.decompress(b) if path.suffix=='.xz' else b)

    def save(self,name,value,purpose):
        dump(DATA/name,value)
        manifest=json.loads((DATA/'MANIFEST.json').read_text())
        b=(DATA/name).read_bytes()
        manifest['files'][name]=dict(sha256=hashlib.sha256(b).hexdigest(),bytes=len(b),encoding='curated JSON',purpose=purpose,sources=self.used,top_level_keys_or_columns=list(value))
        dump(DATA/'MANIFEST.json',manifest);self.used=[]

    def collect(self,step):
        if step=='speed':
            old=json.loads((DATA/'speed.json').read_text())
            rows={mode:performance(self.load('speed/'+mode)) for mode in ('eager','graphs','overlap') if 'speed/'+mode in self.sources}
            p=profile(self.load('speed/profile'))
            self.save('speed.json',dict(reference=old['reference'],reference_provenance=old['reference_provenance'],runs=rows,profile=p),'Generation counts, seconds, peak bytes and step profile; reference has a different memory/timing scope.')
        elif step=='equivalence':
            reports=[];greedy=[]
            for c in CHECKPOINTS:
                for layout in ('full','cached-1','cached-256','generated'):
                    r=parity(self.load(f'parity/{layout}/{c}'));r['layout']=layout;r['checkpoint']=c
                    if layout=='generated':
                        reg=self.load(f'parity/generated/{c}-limits')
                        assert self.used[-1]['source_sha256']==r['limits_sha256']
                        assert reg['saved_unix_time']==r['limits_saved_unix_time']<r['comparison_unix_time']
                        corpus=self.load(f'parity/generated/{c}-corpus')
                        assert self.used[-1]['source_sha256']==reg['corpus_manifest_sha256']
                        r['registration']=select(reg,'saved_unix_time corpus_manifest_sha256 source layouts')
                    reports.append(r)
                g=self.load('greedy/'+c)
                greedy.append(select(g,'checkpoint status median_first_divergence reference_median_first_divergence median_limit position_zero_divergences reference_position_zero_divergences'))
            self.save('equivalence.json',dict(reports=reports,greedy=greedy),'Per-layout aggregate G-TF metrics and reference limits, generated registration timestamps/hashes, greedy medians.')
        elif step=='diagnostics':
            d=self.load('diagnostics')
            self.save('diagnostics.json',dict(sequence_count=len(d['sequences']),logits=d['logits'],sampler=d['sampler']),'Long-context logit buckets and identical-logit sampler distributions (TV, support, entropy).')
        elif step=='robustness':
            rows={k:robustness(self.load('robustness/'+k)) for k in ('swap','stress','prompt','restart')}
            identity=self.load('robustness/overlap') if 'robustness/overlap' in self.sources else None
            self.save('robustness.json',dict(tests=rows,overlap=select(identity,'status rows_identical decisions_identical tokens_counts_identical forced_events runtime_sha256') if identity else None),'Swap identity, long-generation/prompt and restart completion/cleanup; overlap composition-controlled identity.')
        elif step=='negative':
            self.save('negative.json',dict(compiled=performance(self.load('negative/perf')),graphs=performance(self.load('negative/graphs')),parity=parity(self.load('negative/parity')),greedy=select(self.load('negative/greedy'),'status median_first_divergence reference_median_first_divergence median_limit')),'Compile-enabled throughput and failing learned-decider numerical/greedy checks.')
        else:raise ValueError(step)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('step',choices=['speed','equivalence','diagnostics','robustness','negative'])
    p.add_argument('--sources',type=Path,default=ROOT/'scripts/sources.json')
    a=p.parse_args();Collector(json.loads(a.sources.read_text())).collect(a.step)

if __name__=='__main__':main()
