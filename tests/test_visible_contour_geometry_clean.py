import numpy as np

from scripts.audit_visible_contour_geometry_clean import largest_component


def test_largest_component_removes_disconnected_renderer_speckles():
    mask = np.zeros((10, 12), dtype=bool)
    mask[2:6, 3:7] = True
    mask[9, 0:2] = True
    clean = largest_component(mask)
    assert clean.sum() == 16
    assert clean[2:6, 3:7].all()
    assert not clean[9].any()


def test_largest_component_is_empty_for_empty_mask():
    assert not largest_component(np.zeros((4, 4), dtype=bool)).any()
