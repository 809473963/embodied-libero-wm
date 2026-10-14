#!/usr/bin/env python3
"""Persistently advance the finite frozen chain without per-stage approvals."""
import sys,subprocess,time,traceback,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.probes.candidate_utility_oracle import atomic_json
OUT=ROOT/'results/relational_autorun'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'pipeline.json').exists():raise RuntimeError('Existing execution retained')
    rt=dict(status='RUNNING',started_epoch=time.time(),completed_stages=[],confirmation_executed=False)
    stages=[('trajectories',['scripts/collect_relational_autorun.py','trajectories']),('branches',['scripts/collect_relational_autorun.py','branches']),('models',['scripts/evaluate_relational_autorun.py']),('verification_report',['scripts/report_relational_autorun.py'])]
    try:
        for name,args in stages:
            rt['phase']=name;atomic_json(OUT/'pipeline.json',rt)
            subprocess.run([sys.executable,*args],cwd=ROOT,check=True)
            rt['completed_stages'].append(name);atomic_json(OUT/'pipeline.json',rt)
        rt.update(status='COMPLETE',phase='finite_research_chain_complete',finished_epoch=time.time(),report='docs/relational_autorun_audit.md');atomic_json(OUT/'pipeline.json',rt)
    except Exception:
        rt.update(status='BLOCKED',error=traceback.format_exc());atomic_json(OUT/'pipeline.json',rt);raise
if __name__=='__main__':main()
