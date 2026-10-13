#!/usr/bin/env python3
"""CPU regression for curated-data integrity, deterministic outputs and the R1 hook.

R1 uses copied historical CSVs as synthetic test inputs inside a temporary
chapter directory. Those test inputs are never archived or reported as R1.
"""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

sys.dont_write_bytecode=True
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
ROOT=Path(__file__).resolve().parents[1]


def snapshot():
    files=[ROOT/'README.md',ROOT/'APPENDIX.md',*sorted((ROOT/'tables').glob('*')),*sorted((ROOT/'figures').glob('*'))]
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()}


def main():
    for p in ROOT.rglob('*.py'):ast.parse(p.read_text())
    subprocess.run([sys.executable,str(ROOT/'analyze.py')],check=True)
    first=snapshot()
    subprocess.run([sys.executable,str(ROOT/'analyze.py')],check=True)
    assert first==snapshot(),'Nondeterministic generated outputs'
    assert '{{' not in (ROOT/'README.md').read_text()
    print('README, appendix, tables and PDF: byte-identical on consecutive runs')
    # Every named reproduction plan must be inspectable without GPU imports.
    for step in json.loads((ROOT/'data/method.json').read_text())['reproduction']:
        result=subprocess.run([sys.executable,str(ROOT/'scripts/reproduce.py'),step,'--dry-run'],check=True,capture_output=True,text=True)
        assert 'Then extract' in result.stdout
    print('All named reproduction dry-runs: PASS')
    spec=importlib.util.spec_from_file_location('chapter_analysis',ROOT/'analyze.py')
    chapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(chapter)
    import copy
    record=next(r for r in chapter.read('equivalence.json')['reports'] if r['checkpoint']=='real' and r['layout']=='full')
    zero=copy.deepcopy(record)
    zero['summary']['common_decision_mismatches']=0
    zero['summary']['pass_count_mismatches']=0
    for depth in zero['summary']['per_depth']:depth['common_decision_mismatches']=0
    assert chapter.ratio_cells([zero])[:2]==['0.00','0.00']
    zero['noise_limits']['pass_count_mismatches']=0
    for depth in zero['decision_limits']['per_depth']:depth['reference_flips']=0
    assert chapter.ratio_cells([zero])[:2]==['exact','exact']
    print('Ratio labels distinguish zero engine errors from exact agreement: PASS')
    chapter.exact_section()
    with tempfile.TemporaryDirectory(prefix='.check-',dir=ROOT) as temporary:
        original=chapter.DATA;chapter.DATA=Path(temporary)
        try:
            bad=json.loads((original/'exact.json').read_text());bad['tiny']['mlp3']['top1_mismatches']=1
            (chapter.DATA/'exact.json').write_text(json.dumps(bad))
            try:chapter.exact_section()
            except ValueError:pass
            else:raise AssertionError('Non-exact level-0 data accepted')
        finally:chapter.DATA=original
    print('Level 0: exact data accepted, a single mismatch rejected: PASS')
    policy=chapter.read('method.json')['confirmatory']
    with tempfile.TemporaryDirectory(prefix='.check-',dir=ROOT) as temporary:
        tmp=Path(temporary);chapter.DATA=tmp/'data';chapter.TABLES=tmp/'tables'
        chapter.DATA.mkdir();chapter.TABLES.mkdir()
        shutil.copy(ROOT/'data/method.json',chapter.DATA/'method.json')
        assert 'Pending' in chapter.r1_section(policy)
        (chapter.DATA/'r1').mkdir()
        index=dict(complete=False)
        (chapter.DATA/'r1/inputs.json').write_text(json.dumps(index))
        assert 'Pending' in chapter.r1_section(policy)
        # Synthetic complete inputs exercise the actual repository compare API.
        specs=[('reference',1234,'reference-exploratory'),('engine42',42,'engine-seed42'),
               ('engine1234',1234,'engine-seed1234'),('engine2024',2024,'engine-seed42')]
        for label,seed,original in [*specs,('earlier',42,'reference-exploratory')]:
            dest=chapter.DATA/('evaluations/reference-exploratory' if label=='earlier' else 'r1/'+label)
            for ds,n in policy['problems'].items():
                leaf=dest/ds;leaf.mkdir(parents=True)
                shutil.copy(ROOT/f'data/evaluations/{original}/{ds}/detailed_results.csv',leaf/'detailed_results.csv')
                (leaf/'run_status.json').write_text(json.dumps(dict(status='complete',random_seed=seed,expected_samples=n*2,completed_samples=n*2)))
        index=dict(complete=True,frozen_engine_sha256='a'*64,reference=dict(seed=1234,path='r1/reference'),
            engine=[dict(seed=seed,path='r1/'+label,frozen_engine_sha256='a'*64) for label,seed,_ in specs[1:]],deviations=['Synthetic test data only'])
        path=chapter.DATA/'r1/inputs.json';path.write_text(json.dumps(index))
        result=chapter.r1_section(policy)
        assert 'Student-t' in result and 'Exploratory reference' in result and 'equivalence_not_shown' in result
        output=json.loads((chapter.TABLES/'r1-analysis.json').read_text())
        assert output['comparisons'][0]['pooled']['n_problems']==sum(policy['problems'].values())
        index['engine'][0]['seed']=99;path.write_text(json.dumps(index))
        try:chapter.r1_section(policy)
        except AssertionError:pass
        else:raise AssertionError('Wrong R1 seed accepted')
        index['engine'][0]['seed']=42;path.write_text(json.dumps(index))
        csv=chapter.DATA/'r1/engine42/math500/detailed_results.csv'
        lines=csv.read_text().splitlines(True)
        csv.write_text(''.join(lines+lines[1:2]))
        try:chapter.r1_section(policy)
        except ValueError:pass
        else:raise AssertionError('Duplicate R1 sample accepted')
    print('R1: missing/incomplete skip, complete API analysis, wrong-seed and duplicate rejection PASS')
    size=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
    assert size<3_000_000,size
    print(f'Chapter size: {size/1e6:.2f} MB')

if __name__=='__main__':main()
