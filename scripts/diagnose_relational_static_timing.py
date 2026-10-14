import os
os.environ['MUJOCO_GL']='egl';os.environ['PYOPENGL_PLATFORM']='egl'
import sys,json
from pathlib import Path
ROOT=Path('/home/luo/embodied-libero-wm');sys.path.insert(0,str(ROOT));os.environ['LIBERO_CONFIG_PATH']=str(ROOT/'.libero')
import numpy as np,mujoco
from scripts.evaluate_c0_all import make_env
from scripts.inventory_visual_target_masks import render_rgb
r=json.load(open(ROOT/'configs/relational_localization_contract_protocol.json'))['cases'][0]
c=json.load(open(ROOT/'configs/task_grounding_contract.json'))['tasks'][0]
with np.load(ROOT/r['dense']['arrays']) as f:a={k:f[k] for k in ['qpos','qvel','rgb','natural_step','pose','grip_pos']}
e=make_env(ROOT/c['bddl_path']);out=[]
try:
 for hidden in [False,True]:
  if hidden:e.env.visualize(vis_settings={vis:False for vis in e.env._visualizations})
  for t in [0,r['anchor']]:
   for rollback in [False,True]:
    s=e.sim;s.data.qpos[:]=a['qpos'][t];s.data.qvel[:]=a['qvel'][t]
    if rollback and t>0:mujoco.mj_integratePos(s.model._model,s.data.qpos,s.data.qvel,-s.model.opt.timestep)
    s.forward();rgb=render_rgb(e.env);orig=a['rgb'][np.flatnonzero(a['natural_step']==t)[0]]
    ids=[s.model.body_name2id(b) for b in r['dense']['geometry']['bodies'][:3]]
    row=dict(hidden=hidden,step=t,rollback=rollback,equal=bool(np.array_equal(rgb,orig)),mae=float(abs(rgb.astype(float)-orig).mean()),bodydiff=float(abs(s.data.body_xpos[ids]-a['pose'][t,:3,:3]).max()))
    out.append(row)
 print(json.dumps(out,indent=2));(ROOT/'results/relational_localization_contract/timing_diagnostic.json').write_text(json.dumps(out,indent=2)+'\n')
finally:e.close()
