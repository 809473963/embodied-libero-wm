#!/usr/bin/env python3
"""Supervise the frozen Stage1 job, then write its decision report.

No experiment choices are made here. The registered collector and analyzer
remain authoritative. A process lock prevents simultaneous duplicate runs.
"""
from __future__ import annotations

import datetime
import fcntl
import json
import os
import selectors
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.probes.candidate_utility_oracle import atomic_json, sha256


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sync_path(path):
    if path.exists():
        descriptor=os.open(path,os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def durable_checkpoint(directory, progress):
    """Flush a published checkpoint before advancing the runtime record."""
    root=progress.get('root_complete',progress.get('root'))
    if root is None:
        return
    paths=[directory/'arrays'/f'{root}.npz',directory/'access_ledger.json']
    if 'root_complete' in progress:
        paths += [directory/'roots'/f'{root}.json',directory/'manifest.json']
    else:
        start=progress['candidate_branches_complete']-8
        paths += [directory/'groups'/f'{root}_{start:02d}.json',
                  directory/'groups'/f'{root}_{start:02d}.npz']
    for path in paths:
        sync_path(path)
    for path in {p.parent for p in paths}:
        sync_path(path)


def report(directory, runtime):
    p=directory/'metrics.json'
    m=json.loads(p.read_text())
    a=m['primary']
    labels={'A_CANDIDATE_OPPORTUNITY_WITH_USEFUL_T1':'A：候选有机会，T1暂可用',
            'B_CANDIDATE_OPPORTUNITY_BUT_T1_UTILITY_FAILS':'B：候选有机会，T1效用失败',
            'C_INSUFFICIENT_PILOT_CANDIDATE_OPPORTUNITY':'C：pilot候选机会不足',
            'D_COVERAGE_INCONCLUSIVE':'D：覆盖不足，结论未定',
            'S_SIMPLE_GRIPPER_RULE_SATURATES_PILOT_HEADROOM':'S：廉价gripper规则已捕获pilot全部机会'}
    n=a['roots']
    rate=lambda count: f'{100*count/n:.2f}%' if n else 'N/A'
    rows=['# Candidate Utility Oracle：真实候选效用审计','',
          f"结果：**{labels[m['verdict']]}**。仅为calibration pilot，不构成独立新场景确认。",'',
          f"协议 SHA-256：`{m['protocol_sha256']}`。冻结实现提交：`4f5cf66`。",'',
          f"已完成40个注册root和2,560条主候选分支；有效未完成root={n}，此前成功root={m['root_accounting']['ceiling_previously_successful']}。",'',
          '| 方法 | 成功root | 对A0救回 | 伤害 | 净变化 |','|---|---:|---:|---:|---:|',
          f"| A0 + 固定C0 continuation | {a['a0_successes']}/{n} | — | — | — |",
          f"| 集合内outcome oracle | {a['oracle_successes']}/{n} | {a['rescue_roots']} | 0 | +{rate(a['rescue_roots'])} |",
          f"| H8 T1-selected | {a['selected_successes']}/{n} | {a['captured_rescue_roots']} | {a['harm_roots']} | {rate(a['net_success_gain_roots'])} |",
          f"| c0_binarized固定规则 | {m['c0_binarized']['selected_successes']}/{n} | {m['c0_binarized']['captured_rescue_roots']} | {m['c0_binarized']['harm_roots']} | {rate(m['c0_binarized']['net_success_gain_roots'])} |",'',
          f"rescue覆盖{m['rescue_task_count']}个任务；T1 capture fraction={a['capture_fraction']}，override={a['override_roots']}个root，mean regret={a['mean_regret']}。",'',
          f"T1净success变化的95%描述性区间：{m['ci95']['net_success_gain_rate']}；oracle rescue区间：{m['ci95']['oracle_rescue_rate']}。按任务分层重采样root，不把64个候选当作独立样本。每任务仅4个root，区间不覆盖训练seed、任务选择或全部场景分布的不确定性。",'',
          '| Gate | 结果 |','|---|---|']
    rows += [f'| {k} | {v} |' for k,v in m['gates'].items()]
    rows += ['','| 任务 | 有效root | A0成功 | Oracle成功 | T1成功 | Rescue | Harm |','|---|---:|---:|---:|---:|---:|---:|']
    rows += [f"| {t} | {v['roots']} | {v['a0_successes']} | {v['oracle_successes']} | {v['selected_successes']} | {v['rescue_roots']} | {v['harm_roots']} |" for t,v in m['per_task'].items()]
    rows += ['','| 子集 | Oracle成功 | T1成功 | Rescue | T1 harm |','|---|---:|---:|---:|---:|---:|']
    rows += [f"| {k} | {v['oracle_successes']} | {v['selected_successes']} | {v['rescue_roots']} | {v['harm_roots']} |" for k,v in m['subsets'].items()]
    rows += ['',f"下一动作：`{m['next_step']}`。本次收集结束；没有自动启动下一阶段训练或learned controller。",'',
             '所有候选从fresh env及同seed/init/prefix起步，H8后使用同一冻结C0至总300步。官方success首次命中为primary；最终On与H8几何为诊断。G2身份从初始scene确定并保持。A0和预声明T1/rescue分支通过重复动作及结果检查。', '',
             '未打开HDF5；官方init容器只将31–34行构造为数值数组。旧协议和运行授权保持原样；没有训练encoder、value或WM。', '',
             f"各次运行已记录耗时合计：{runtime.get('prior_observed_elapsed_seconds',0)+runtime['elapsed_seconds']:.1f}秒；中断期间不计入。技术preflight另见结果。",'',
             '[完整指标](../results/candidate_utility_oracle/metrics.json)、[决策](../results/candidate_utility_oracle/decision.json)、[preflight](../results/candidate_utility_oracle/preflight.json)、[访问记录](../results/candidate_utility_oracle/access_ledger.json)、[运行状态](../results/candidate_utility_oracle/runtime.json)。','']
    path=ROOT/'docs/candidate_utility_oracle_audit.md'
    temp=path.with_suffix('.md.part')
    temp.write_text('\n'.join(rows))
    temp.replace(path)


def main():
    directory=ROOT/'results/candidate_utility_oracle'
    directory.mkdir(parents=True,exist_ok=True)
    lock=open(directory/'run.lock','a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.monotonic()
    prior={}
    history=None
    if (directory/'runtime.json').exists():
        prior=json.loads((directory/'runtime.json').read_text())
        history=directory/'run_history'/f"{prior['pid']}_{int(time.time())}.json"
        atomic_json(history,prior)
        sync_path(history)
        sync_path(history.parent)
    status={'pid':os.getpid(),'started_at':now(),'status':'RUNNING','phase':'collection',
            'protocol_sha256':sha256(ROOT/'configs/candidate_utility_oracle_protocol.json'),
            'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'prior_run_snapshot':str(history) if history else None,
            'prior_observed_elapsed_seconds':prior.get('prior_observed_elapsed_seconds',0)+prior.get('elapsed_seconds',0)}
    if prior.get('progress'):
        status['last_prior_progress']=prior['progress']
    def save():
        status['updated_at']=now()
        status['elapsed_seconds']=time.monotonic()-started
        atomic_json(directory/'runtime.json',status)
        sync_path(directory/'runtime.json')
        sync_path(directory)
    save()
    try:
        commands=[[sys.executable,'scripts/collect_candidate_utility_oracle.py','--mode','collect'],
                  [sys.executable,'scripts/analyze_candidate_utility_oracle.py']]
        for command in commands:
            status['phase']='collection' if '--mode' in command else 'analysis'
            save()
            with subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,bufsize=0) as process:
                status['child_pid']=process.pid
                save()
                selector=selectors.DefaultSelector()
                selector.register(process.stdout,selectors.EVENT_READ)
                pending=b''
                while True:
                    if not selector.select(timeout=30):
                        save()
                        continue
                    block=os.read(process.stdout.fileno(),65536)
                    if not block:
                        if pending:
                            print(pending.decode(errors='replace'),end='',flush=True)
                        break
                    lines=(pending+block).split(b'\n')
                    pending=lines.pop()
                    for raw in lines:
                        line=raw.decode(errors='replace')
                        print(line,flush=True)
                        try:
                            progress=json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(progress,dict) and ('root' in progress or 'root_complete' in progress):
                            durable_checkpoint(directory,progress)
                            status['progress']=progress
                            save()
                selector.close()
                if process.wait()!=0:
                    raise RuntimeError(f'{status["phase"]} failed with exit {process.returncode}; see collection_blocker.json/driver.log')
        status['phase']='report'
        save()
        report(directory,status)
        status['status']='COMPLETE'
        status['phase']='finished'
        status['report']='docs/candidate_utility_oracle_audit.md'
        for path in (directory/'metrics.json',directory/'decision.json',ROOT/status['report']):
            sync_path(path)
            sync_path(path.parent)
        save()
    except Exception as error:
        status['status']='TECHNICAL_BLOCKED'
        status['error']=str(error)
        save()
        raise


if __name__=='__main__':
    main()
