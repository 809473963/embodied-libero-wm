#!/usr/bin/env python3
import json,sys,subprocess,traceback,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.probes.candidate_utility_oracle import atomic_json
OUT=ROOT/'results/relational_event_h24'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'pipeline.json').exists() or (OUT/'runtime.json').exists():raise RuntimeError('Existing run retained')
    state=dict(status='RUNNING',phase='preflight_and_collection',started_epoch=time.time(),training=False);atomic_json(OUT/'pipeline.json',state)
    try:
        subprocess.run([sys.executable,'scripts/run_relational_event_h24.py'],cwd=ROOT,check=True)
        state['phase']='verification_and_event_coverage';atomic_json(OUT/'pipeline.json',state)
        subprocess.run([sys.executable,'scripts/analyze_relational_event_h24.py'],cwd=ROOT,check=True)
        state.update(status='COMPLETE',phase='awaiting_scientific_review',finished_epoch=time.time());atomic_json(OUT/'pipeline.json',state)
    except Exception:
        state.update(status='BLOCKED',error=traceback.format_exc());atomic_json(OUT/'pipeline.json',state);raise

if __name__=='__main__':main()
