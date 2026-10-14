#!/usr/bin/env python3
"""Bounded background sequence: record, verify, summarize; never start training."""
import json,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/relational_coverage_expansion'

def save(state):
    tmp=OUT/'pipeline.tmp';tmp.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n')
    tmp.replace(OUT/'pipeline.json')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'pipeline.json').exists() or (OUT/'runtime.json').exists():
        raise RuntimeError('Existing run retained; no automatic restart')
    state=dict(status='RUNNING',phase='recording',started_epoch=time.time(),training=False)
    save(state)
    try:
        subprocess.run([sys.executable,'scripts/run_relational_coverage_expansion.py'],cwd=ROOT,check=True)
        state['phase']='verification_and_coverage';save(state)
        subprocess.run([sys.executable,'scripts/finalize_relational_coverage_expansion.py'],cwd=ROOT,check=True)
        state.update(status='COMPLETE',phase='awaiting_scientific_coverage_review',finished_epoch=time.time());save(state)
    except Exception:
        state.update(status='BLOCKED',error=traceback.format_exc(),scientific_conclusion='not assigned');save(state)
        raise

if __name__=='__main__':main()
