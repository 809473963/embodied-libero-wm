#!/usr/bin/env python3
import sys,json,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.probes.candidate_utility_oracle import atomic_json
P=ROOT/'results/relational_localization_cv'
def main():
 P.mkdir(exist_ok=True);path=P/'pipeline.json'
 if path.exists():raise RuntimeError('Existing pipeline retained')
 rt=dict(status='RUNNING',completed_stages=[],started_epoch=time.time(),confirmation_opened=False,new_development_opened=False);atomic_json(path,rt)
 for stage,script in [('collection','collect_relational_localization.py'),('training','train_relational_localization_cv.py'),('verification','report_relational_localization_cv.py')]:
  rt['phase']=stage;atomic_json(path,rt)
  run=subprocess.run([sys.executable,str(ROOT/'scripts'/script)],cwd=ROOT)
  if run.returncode:
   rt.update(status='BLOCKED',returncode=run.returncode);atomic_json(path,rt);return
  rt['completed_stages'].append(stage);atomic_json(path,rt)
 rt.update(status='COMPLETE',phase='verified_and_reported',finished_epoch=time.time(),report='docs/relational_localization_cv_audit.md');atomic_json(path,rt)
if __name__=='__main__':main()
