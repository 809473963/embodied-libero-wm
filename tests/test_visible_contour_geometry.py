import numpy as np
import cv2
from scripts.audit_visible_contour_geometry import fit_shape,mesh_projection,perturb,bounds

def test_mask_geometry_does_not_need_truth_depth():
 v=np.array([[x,y,z] for x in [-.05,.05] for y in [-.04,.04] for z in [-.02,.02]])
 K=np.array([[150.,0,64],[0,150.,64],[0,0,1.]])
 truth=np.array([.03,-.02,.7]);uv,_=mesh_projection(v,np.eye(3),truth,K,np.eye(3))
 mask=np.zeros((128,128),np.uint8);cv2.fillConvexPoly(mask,cv2.convexHull(uv.astype(np.float32)).astype(np.int32),1)
 p=fit_shape(mask.astype(bool),v,np.eye(3),K,np.eye(4),'bbox')
 assert p['positive_depth'] and np.linalg.norm(p['xyz']-truth)<.04
 assert fit_shape(np.zeros_like(mask),v,np.eye(3),K,np.eye(4)) is None

def test_pixel_perturbations_preserve_unknown_and_do_not_mutate():
 m=np.zeros((128,128),bool);m[40:60,50:70]=True;copy=m.copy()
 assert perturb(m,'erode1').sum()<m.sum()<perturb(m,'dilate1').sum()
 assert perturb(m,'res64').shape==m.shape
 np.testing.assert_array_equal(copy,m);np.testing.assert_allclose(bounds(m),[49.5,39.5,69.5,59.5])
