#!/usr/bin/env python3
"""CPU regression for curated-data integrity, deterministic outputs and the R1 hook.

R1 uses a temporary copy of the compact confirmatory inputs to exercise the
comparison adapter and its validation failures.
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
SCRATCH=ROOT.parents[1]/'data/tmp/wp25/checks/1-engine'


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
        if step in ('r2_speed','memory_ab'):
            assert 'curate_wp25.py' in result.stdout
        else:
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
    SCRATCH.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='check-',dir=SCRATCH) as temporary:
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
    with tempfile.TemporaryDirectory(prefix='check-',dir=SCRATCH) as temporary:
        tmp=Path(temporary);chapter.DATA=tmp/'data';chapter.TABLES=tmp/'tables'
        chapter.DATA.mkdir();chapter.TABLES.mkdir()
        shutil.copy(ROOT/'data/method.json',chapter.DATA/'method.json')
        assert 'Pending' in chapter.r1_section(policy)
        (chapter.DATA/'r1').mkdir()
        index=dict(complete=False)
        (chapter.DATA/'r1/runs.json').write_text(json.dumps(index))
        assert 'Pending' in chapter.r1_section(policy)
        shutil.copy(ROOT/'data/r1/samples.csv',chapter.DATA/'r1/samples.csv')
        shutil.copy(ROOT/'data/r1/runs.json',chapter.DATA/'r1/runs.json')
        path=chapter.DATA/'r1/runs.json';index=json.loads(path.read_text())
        result=chapter.r1_section(policy)
        assert 'ref. 1234' in result and 'Engine' in result and 'Ref. 42' not in result
        assert chapter.R1_FACTS['r1_verdict']=='equivalence not shown'
        output=json.loads((chapter.TABLES/'r1-analysis.json').read_text())
        assert output['comparisons'][0]['pooled']['n_problems']==sum(policy['problems'].values())
        index_bad=json.loads(json.dumps(index));index_bad['engine'][0]['seed']=99;path.write_text(json.dumps(index_bad))
        try:chapter.r1_section(policy)
        except AssertionError:pass
        else:raise AssertionError('Wrong R1 seed accepted')
        path.write_text(json.dumps(index))
        bad=json.loads(json.dumps(index));bad['runs'][1]['groups'][0]['settings']['temperature']=0.7
        path.write_text(json.dumps(bad))
        try:chapter.r1_section(policy)
        except ValueError:pass
        else:raise AssertionError('Settings differing from the protocol accepted')
        path.write_text(json.dumps(index))
        csv=chapter.DATA/'r1/samples.csv';lines=csv.read_text().splitlines(True)
        csv.write_text(''.join(lines+[lines[1]]))
        try:chapter.r1_section(policy)
        except ValueError:pass
        else:raise AssertionError('Duplicate R1 sample accepted')
    print('R1: missing/incomplete skip, compact API analysis, wrong-seed, wrong-settings and duplicate rejection PASS')
    size=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
    assert size<3_000_000,size
    print(f'Chapter size: {size/1e6:.2f} MB')

if __name__=='__main__':main()
