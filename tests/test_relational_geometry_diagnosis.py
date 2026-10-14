import json
import numpy as np
from scripts.diagnose_relational_geometry_errors import ROOT,world_from_head,macro
from src.probes.camera_projection_contract import independent_projection

def test_inverse_uses_upright_pixel_convention_and_camera_depth():
 c=json.loads((ROOT/'results/camera_projection_contract/camera_inventory.json').read_text())['camera_reference']
 xyz=np.array([[[.1,.1,.95],[-.1,.2,1.1],[0.,.15,.9]]])
 _,uv,depth=independent_projection(xyz,np.array(c['camera_position_world']),np.array(c['camera_rotation_world']),c['fovy_degrees'])
 head=np.concatenate([uv.reshape(-1,6),depth],1)
 np.testing.assert_allclose(world_from_head(head,np.array(c['world_to_raw_pixel'])),xyz,atol=1e-9,rtol=0)

def test_macro_balances_scenes_not_anchor_counts():
 assert macro(np.array([2.,2.,2.,10.]),np.array(['a','a','a','b']))==6.
