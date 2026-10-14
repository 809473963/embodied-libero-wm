#!/usr/bin/env python3
"""Static reviewed figures and offline case gallery, no flowchart or new inference."""
import sys,json,html
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw,ImageFont
from scripts.diagnose_multi_object_perception import inputs,read_npz,OUT,CV,NAMES,project
from src.probes.candidate_utility_oracle import atomic_json

def main():
 d,n,meta=inputs();roots=n['roots'];unique=sorted(set(roots));truth=n['positions'];shared=read_npz(CV/'oof_shared_ensemble.npz')['nodes'];ind=read_npz(CV/'oof_independent_ensemble.npz')['nodes'];M=np.array(json.loads((ROOT/'results/camera_projection_contract/camera_inventory.json').read_text())['camera_reference']['world_to_raw_pixel'])
 out=OUT/'figures';out.mkdir(exist_ok=True)
 err=[np.linalg.norm(p-truth,axis=2)*1000 for p in [ind,shared]]
 fig,axes=plt.subplots(1,3,figsize=(15,7),layout='constrained');matrices=[np.array([e[roots==r,:5].mean(0) for r in unique]) for e in err]
 for ax,mat,title in zip(axes,[*matrices,matrices[1]-matrices[0]],['Independent: mm','Shared: mm','Shared minus independent: mm']):
  delta='minus' in title;im=ax.imshow(mat,cmap='coolwarm' if delta else 'magma',vmin=-60 if delta else 0,vmax=60 if delta else 450,aspect='auto');ax.set_xticks(range(5),NAMES[:5],rotation=45);ax.set_yticks(range(13),unique);ax.set_title(title)
  for i in range(13):
   for j in range(5):ax.text(j,i,f'{mat[i,j]:.0f}',ha='center',va='center',fontsize=8,color='black' if delta else 'white')
  fig.colorbar(im,ax=ax,shrink=.7)
 fig.savefig(out/'scene_object_errors.png',dpi=140);plt.close(fig)
 fig=plt.figure(figsize=(14,6));ax=fig.add_subplot(121,projection='3d');bx=fig.add_subplot(122)
 colors=plt.get_cmap('tab20')
 for k,r in enumerate(unique):
  ii=np.flatnonzero(roots==r);ii=ii[np.linspace(0,len(ii)-1,min(30,len(ii))).astype(int)];c=colors(k)
  ax.scatter(*truth[ii,1].T,c=[c],s=10);ax.scatter(*shared[ii,1].T,c=[c],marker='x',s=10)
  gt=truth[roots==r,1].mean(0);pr=shared[roots==r,1].mean(0);bx.plot([gt[0],pr[0]],[gt[1],pr[1]],c=c);bx.scatter(gt[0],gt[1],c=[c],s=35);bx.scatter(pr[0],pr[1],c=[c],marker='x',s=35);bx.annotate(r,(gt[0],gt[1]),fontsize=7)
 ax.set(xlabel='world X (m)',ylabel='world Y (m)',zlabel='world Z (m)',title='Bowl2: GT dots / shared crosses; fixed sampled frames')
 bx.set(xlabel='world X (m)',ylabel='world Y (m)',title='Bowl2 scene means: GT dot -> predicted cross');bx.set_aspect('equal');fig.tight_layout();fig.savefig(out/'bowl2_world_positions.png',dpi=150);plt.close(fig)
 ident=json.loads((OUT/'identity_diagnosis.json').read_text())['shared_ensemble']['scenes'];fig,ax=plt.subplots(figsize=(12,5));x=np.arange(13)
 ax.bar(x-.2,[ident[r]['bowl2_original_mm'] for r in unique],.4,label='Original');ax.bar(x+.2,[ident[r]['bowl2_oracle_mm'] for r in unique],.4,label='GT-assisted best swap');ax.set_xticks(x,unique,rotation=45,ha='right');ax.set_ylabel('Bowl2 mean error (mm)');ax.legend();fig.tight_layout();fig.savefig(out/'identity_oracle.png',dpi=140);plt.close(fig)
 font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14);cases=[]
 for r in unique:
  ix=np.flatnonzero(roots==r);order=ix[np.argsort(err[1][ix,1])]
  choices=[('worst',int(order[-1])),('median',int(order[len(order)//2])),('initial',int(ix[[meta['rows'][j]['step']==0 for j in ix]][0]))]
  for label,i in choices:
   panels=[]
   for title,p,color in [('GT body origins',truth,(40,255,70)),('Shared predictions',shared,(255,80,60)),('Independent predictions',ind,(60,200,255))]:
    tile=Image.fromarray(d['images'][i]).resize((384,384),Image.Resampling.NEAREST);draw=ImageDraw.Draw(tile);uv,depth=project(p[i,:5],M);gtuv,_=project(truth[i,:5],M)
    for obj in range(5):
     gu,gv=(gtuv[obj]-.5)/2*6;u,v=(uv[obj]-.5)/2*6
     if depth[obj]>0:
      cx,cy=float(np.clip(u,5,378)),float(np.clip(v,28,378));draw.line((gu,gv,u,v),fill=color,width=1);draw.ellipse((cx-4,cy-4,cx+4,cy+4),outline=color,width=2);draw.text((cx+4,cy),str(obj),font=font,fill=color)
    draw.rectangle((0,0,384,24),fill='black');draw.text((4,3),title,font=font,fill='white');panels.append(tile)
   canvas=Image.new('RGB',(1152,420),'black')
   for j,tile in enumerate(panels):canvas.paste(tile,(j*384,30))
   draw=ImageDraw.Draw(canvas);caption=f"{r} / {label} / row {i} / step {meta['rows'][i]['step']} / bowl2 shared {err[1][i,1]:.1f}mm / independent {err[0][i,1]:.1f}mm";draw.text((4,5),caption,font=font,fill='white')
   name=f'case_{r}_{label}.png';canvas.save(out/name);cases.append(dict(row=i,root=r,kind=label,step=meta['rows'][i]['step'],source=meta['rows'][i]['source'],image='figures/'+name,caption=caption))
 atomic_json(OUT/'visual_case_manifest.json',dict(cases=cases,selection='per scene max/median bowl2 shared error and initial frame; descriptive outcome-selected cases, not representative sample',projection='calibrated body-origin projections, not visible pixel centers; outside-frame predictions clamped to border for marker display',object_indices=NAMES[:5]))
 options=''.join(f'<option value="{i}">{html.escape(c["caption"])}</option>' for i,c in enumerate(cases))
 page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>多对象感知错误案例</title><style>body{font:17px system-ui;background:#f5f6f8;color:#182331;margin:30px auto;max-width:1250px;padding:16px}img{max-width:100%;background:white}select{width:100%;padding:12px}p{line-height:1.7}a{color:#1459b3}</style><h1>多对象感知：真实位置与预测位置</h1><p>仅旧13场景OOF预测。图中标记是三维body原点投影，不是肉眼可见质心；连线显示真实与预测投影差异。0=碗1，1=碗2，2=盘，3=cookies，4=ramekin。越界标记显示在边缘，不能视为正确落在图像中。每场景展示最大误差、中位误差与初始帧；案例选择不代表总体频率。</p><select id="s">OPTIONS</select><p id="cap"></p><img id="case"><h2>分对象、分场景误差</h2><img src="figures/scene_object_errors.png"><h2>第二只碗的世界坐标</h2><img src="figures/bowl2_world_positions.png"><h2>身份交换上限（使用答案，不可部署）</h2><img src="figures/identity_oracle.png"><script>const cases=CASES;function show(){let c=cases[+document.getElementById('s').value];document.getElementById('case').src=c.image;document.getElementById('cap').textContent=c.caption;}document.getElementById('s').onchange=show;show();</script></html>'''.replace('OPTIONS',options).replace('CASES',json.dumps(cases,ensure_ascii=False))
 (OUT/'index.html').write_text(page)
 print('Figures and',len(cases),'cases complete')
if __name__=='__main__':main()
