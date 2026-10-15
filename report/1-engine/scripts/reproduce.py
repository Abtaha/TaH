#!/usr/bin/env python3
"""Named GPU experiments followed by CPU extraction into the chapter's data/.
No arguments list steps. --dry-run prints exact commands without executing them.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
sys.dont_write_bytecode=True
from curate import Collector, dump, CHECKPOINTS

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parents[1]
PY=REPO/'.venv-tah/bin/python'
SCRIPTS=ROOT/'scripts'
DATA=ROOT/'data'


def plan(step,out):
    commands=[];sources={}
    def add(label,args,gpu=True,expected_failure=False):
        commands.append(dict(label=label,args=[str(PY),*map(str,args)],gpu=gpu,expected_failure=expected_failure))
    def check(label,gate,*flags):
        dest=out/(label+'.json')
        add(label,[SCRIPTS/'validation_cli.py','engine_check','--gate',gate,'--output',dest,*flags])
        return str(dest)
    def generated(c,compiled=False):
        dest=out/('compiled' if compiled else 'generated')
        for phase in ('generate','noise','compare'):
            add(f'{dest.name}-{c}-{phase}',[SCRIPTS/'validation_cli.py','generated_gtf','--checkpoint',c,'--root',dest,'--phase',phase,*(['--compile'] if compiled else [])],gpu=phase!='compare',expected_failure=compiled and phase=='compare')
        return str(dest/f'tf-{c}-expanded.json'),str(dest/'gtf-limits'/f'{c}.json'),str(dest/'gtf-corpus'/c/'manifest.json')
    if step=='fixtures':
        args=['-m','tah.engine.generate_fixtures','--profile','full','--seed','20261012','--output',out/'fixtures','--model-output',out/'fixture-models']
        add('prepare',[*args,'--prepare-only'],False)
        for c in CHECKPOINTS:add('fixture-'+c,[*args,'--device','cuda','--checkpoint',c])
        add('fixture-limits',[SCRIPTS/'export_fixture_limits.py'],False)
    elif step=='reference':
        add('reference',[SCRIPTS/'reference_perf.py',out/'reference-perf'])
    elif step=='speed':
        for mode,flags in [('eager',['--no-cuda-graphs']),('graphs',['--cuda-graphs']),('overlap',['--cuda-graphs','--overlap'])]:
            sources['speed/'+mode]=check('speed-'+mode,'perf','--batch-size','64','--max-new-tokens','4096','--no-compile',*flags)
        sources['speed/profile']=check('profile','profile','--batch-size','64','--cuda-graphs','--no-compile')
    elif step=='equivalence':
        for c in CHECKPOINTS:
            add('full-'+c,['-m','pytest','-q','-o','addopts=','-m','gpu and slow','tah/engine/tests/test_gtf.py','-k',c])
            sources['parity/full/'+c]=str(out/'full'/f'{c}.json')
            for chunk in (1,256):
                add(f'cached-{chunk}-{c}',[SCRIPTS/'cached_validation.py','--checkpoint',c,'--chunk',str(chunk)],expected_failure=c=='mlp1' and chunk==1)
                sources[f'parity/cached-{chunk}/{c}']=str(out/f'cached-{chunk}'/f'{c}.json')
            report,limits,corpus=generated(c)
            for suffix,path in [('',report),('-limits',limits),('-corpus',corpus)]:sources[f'parity/generated/{c}{suffix}']=path
            sources['greedy/'+c]=check('greedy-'+c,'gen','--checkpoint',c,'--no-compile')
    elif step=='diagnostics':
        sources['diagnostics']=str(out/'longctx.json')
        add('longctx',[SCRIPTS/'longctx.py','--output',sources['diagnostics']])
    elif step=='robustness':
        for key,gate,cap in [('swap','swap',4096),('stress','robust',16384),('prompt','long-prompt',8)]:
            sources['robustness/'+key]=check('robustness-'+key,gate,'--batch-size','64','--max-new-tokens',str(cap),'--cuda-graphs','--no-compile')
        sources['robustness/overlap']=check('overlap-identity','overlap-identity')
        add('restart',['-m','pytest','-q','-s','-o','addopts=','tah/engine/tests/test_engine_gpu.py::test_cuda_overflow_restart_repeated_completion_and_cleanup','--basetemp',out/'restart-test'])
        sources['robustness/restart']=str(out/'restart.json')
    elif step=='negative':
        sources['negative/graphs']=check('negative-graphs','perf','--batch-size','64','--no-compile')
        sources['negative/perf']=check('negative-compiled','perf','--batch-size','64','--compile','--graph-batch-sizes','54,55,58,60,64')
        report,_,_=generated('mlp3',True);sources['negative/parity']=report
        sources['negative/greedy']=check('negative-greedy','gen','--checkpoint','mlp3','--compile')
        commands[-1]['expected_failure']=True
    elif step=='exact':
        # CPU only. The tests write their reports to data/engine/wp3/cpu-pytest/.
        for label,test in [('exact-model','test_model.py'),('exact-probes','test_cpu_probes.py')]:
            add(label,['-m','pytest','-q','-o','addopts=','tah/engine/tests/'+test],gpu=False)
        sources.update({k:v for k,v in json.loads((SCRIPTS/'sources.json').read_text()).items() if k.startswith('exact/')})
    elif step=='accuracy':
        for backend,seeds in [('tah',[1234]),('tah_engine',[42,1234,2024])]:
            for seed in seeds:
                dest=out/f'r1-{backend}-{seed}'
                add(dest.name,['script/tah/evaluation/run_protocol.py','run','--config','script/tah/recipes/qwen3_1.7/protocol_wacc.yaml','--model_path','models/TaH-plus-1.7B','--backend',backend,'--seed',seed,'--output_root',dest])
                add('merge-'+dest.name,['script/tah/evaluation/run_protocol.py','merge','--output_root',dest],False)
    else:raise ValueError(step)
    return commands,sources


def source_inventory():
    return {str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((REPO/'tah/engine').glob('*.py'))}


def accuracy_inputs(out,inventory,commit):
    sys.path.insert(0,str(SCRIPTS))
    try:
        import curate_r1
    finally:
        sys.path.pop(0)
    digest=hashlib.sha256(json.dumps(inventory,sort_keys=True).encode()).hexdigest()
    specs=[]
    for name,backend,seed in [('reference-seed1234','tah',1234),('engine-seed42','tah_engine',42),
                              ('engine-seed1234','tah_engine',1234),('engine-seed2024','tah_engine',2024)]:
        specs.append((name,backend,seed,out/f'r1-{backend}-{seed}','protocol'))
    sample_bytes,index,sources,rows=curate_r1.collect(specs,inventory,digest,commit,direct_2024=False)
    count,file_count=curate_r1.store_compact_archive(sample_bytes,index,sources,rows,drop_legacy=True)
    dump(out/'r1-index.json',index)
    print(f'Imported {count} confirmatory samples in {file_count} chapter data files.')


def main():
    policy=json.loads((DATA/'method.json').read_text())['reproduction']
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step',nargs='?',choices=list(policy))
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--output-root',type=Path,default=ROOT/'reproduction')
    parser.add_argument('--gpu',default='0')
    args=parser.parse_args()
    if not args.step:
        for name,item in policy.items():print(f"{name}: {item['runtime']}; {item['device']}; "+('reproduction/fixtures/' if name=='fixtures' else 'data/r1/samples.csv, data/r1/runs.json' if name=='accuracy' else 'data/speed.json' if name=='reference' else f'data/{name}.json'))
        print('Use STEP --dry-run for exact commands; run fixtures before equivalence/negative. No work executed.')
        return
    out=args.output_root.resolve()
    if out==DATA or out.is_relative_to(DATA):parser.error('Keep full experiment outputs outside curated data/')
    commands,sources=plan(args.step,out)
    for item in commands:print(shlex.join(item['args']),flush=True)
    print('Then extract '+args.step+' metrics, update data/MANIFEST.json, and run analyze.py.',flush=True)
    if args.dry_run:return
    out.mkdir(parents=True,exist_ok=True)
    if args.step in ('equivalence','negative') and not (out/'fixture-limits/tolerances.json').exists():
        parser.error('Run fixtures first using the same --output-root')
    if args.step=='exact':
        need=['data/engine/fixtures/cpu-tiny/manifest.json','data/engine/fixture-models/tiny/label3','data/engine/fixture-models/full/label3','models/TaH-plus-1.7B']
        absent=[x for x in need if not (REPO/x).exists()]
        if absent:parser.error('Missing prerequisites (tests would skip silently): '+', '.join(absent))
    started=time.time()
    inventory=source_inventory()
    if args.step=='accuracy':
        freeze=out/'r1-engine-freeze.json'
        if freeze.exists() and json.loads(freeze.read_text())!=inventory:raise ValueError('Engine changed since R1 began; do not mix versions')
        dump(freeze,inventory)
    env=dict(os.environ,PYTHONPATH=str(REPO),PYTHONDONTWRITEBYTECODE='1',ENGINE_REPORT_OUTPUT=str(out),TAH_GTF_FIXTURES=str(out/'fixtures'),TAH_GTF_TOLERANCES=str(out/'fixture-limits/tolerances.json'),TAH_GTF_RESULTS=str(out/'full'))
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    dump(out/(args.step+'-commands.json'),dict(commands=commands,engine_sources=inventory,git_commit=commit,started_utc=datetime.now(timezone.utc).isoformat()))
    for item in commands:
        env['CUDA_VISIBLE_DEVICES']=args.gpu if item['gpu'] else ''
        log=out/(item['label']+'.log')
        with log.open('w') as f:result=subprocess.run(item['args'],cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT)
        if result.returncode and not (item['expected_failure'] and result.returncode==1):
            raise RuntimeError(f"{item['label']} exited {result.returncode}; inspect {log}")
        if item['label']=='restart':
            line=next(x for x in log.read_text().splitlines() if 'Forced restart CUDA report:' in x)
            value=json.loads(line.split('Forced restart CUDA report:',1)[1]);value['status']='PASS';dump(out/'restart.json',value)
    if inventory!=source_inventory():raise ValueError('Engine changed during the experiment; outputs retained but not imported')
    if args.step=='exact':
        stale=[v for k,v in sources.items() if k not in ('exact/tests','exact/fixtures') and not k.startswith('exact/config/') and (REPO/v).stat().st_mtime<started]
        if stale:raise ValueError('Reports not rewritten by this run (tests skipped?): '+', '.join(stale))
    if args.step=='accuracy':accuracy_inputs(out,inventory,commit)
    elif args.step not in ('fixtures','reference'):
        dump(out/(args.step+'-sources.json'),sources);Collector(sources).collect(args.step)
    subprocess.run([str(PY),str(ROOT/'analyze.py')],env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),check=True)

if __name__=='__main__':main()
