import json
from pathlib import Path

import numpy as np
import pytest

from src.probes.continuation_utility_observability import (
    IDS, MAX_FRAMES, RAW_KEYS, check_inputs, choose_threshold, decisions,
    fit_normalization, loss_function, make_readout, normalize, select, validate_raw,
)


def test_forbidden_outcome_and_privileged_fields_are_rejected():
    inputs = {'sequence': np.zeros((2, 17, 532)), 'lengths': np.array([2, 3]),
              'language': np.zeros((2, 512)), 'budget': np.ones((2, 1))}
    check_inputs(inputs)
    for forbidden in ('success', 'first_success_step', 'final300', 'oracle_winner', 'G2_ref', 'init_id'):
        with pytest.raises(ValueError, match='whitelist'):
            check_inputs({**inputs, forbidden: np.zeros(2)})
    with pytest.raises(ValueError, match='whitelist'):
        validate_raw({**{k: np.zeros(1) for k in RAW_KEYS}, 'final_success': np.zeros(1)})


def test_fit_normalization_excludes_development_and_padding():
    x = np.full((2, 32, 17, 1), 100000., np.float32)
    x[0, :, :2, 0] = [1., 3.]
    lengths = np.full((2, 32), 2)
    mean, std = fit_normalization(x, lengths, np.array([True, False]))
    assert mean[0] == 2 and std[0] == 1
    out = normalize(x, lengths, mean, std)
    assert not out[:, :, 2:].any()
    assert out[0, 0, :2, 0].tolist() == [-1, 1]


def test_selection_preserves_a0_on_ties_threshold_and_disagreement():
    p = np.full((3, 32), .2)
    p[0, 2], p[1, 2], p[2, 2] = .3, .2001, .4
    votes = np.full((3, 32), 3)
    votes[2, 2] = 1
    assert select(p, .01, votes).tolist() == [2, 0, 0]
    assert select(np.ones((3, 32))).tolist() == [0, 0, 0]


def test_decision_threshold_accounts_for_rescue_and_harm():
    y = np.zeros((3, 32), bool)
    y[0, 2], y[1, 0], y[2, 0] = True, True, True
    p = np.full((3, 32), .2)
    p[:, 2] = [.5, .3, .2]
    raw = decisions(y, select(p))
    assert raw['captured_rescue_roots'] == raw['harm_roots'] == 1
    threshold, grid = choose_threshold(p, y, [0., .15, 1.])
    assert threshold == .15
    chosen = decisions(y, select(p, threshold))
    assert chosen['captured_rescue_roots'] == 1 and chosen['harm_roots'] == 0
    assert len(grid) == 3


def test_scene_split_covers_all_roots_without_candidate_overlap():
    root = Path(__file__).resolve().parents[1]
    split = json.loads((root / 'configs/continuation_utility_observability_split.json').read_text())
    registered = json.loads((root / 'configs/candidate_utility_oracle_roots.json').read_text())['roots']
    assert not set(split['fit_roots']) & set(split['development_roots'])
    assert set(split['fit_roots'] + split['development_roots']) == {r['root_id'] for r in registered}
    assert all((r['root_id'] in split['fit_roots']) == (r['init_index'] < 34) for r in registered)
    assert len(IDS) == 32


def test_reader_ignores_padding_and_rejects_labels():
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(11)
    model = make_readout(5).eval()
    inputs = {'sequence': torch.randn(2, MAX_FRAMES, 5), 'lengths': torch.tensor([2, 4]),
              'language': torch.randn(2, 512), 'budget': torch.ones(2, 1)}
    with torch.no_grad():
        expected = model(**inputs)
        inputs['sequence'][0, 2:] = 99999
        inputs['sequence'][1, 4:] = -99999
        assert torch.equal(expected, model(**inputs))
    with pytest.raises(ValueError, match='whitelist'):
        model(**inputs, success=torch.ones(2))


def test_pair_loss_prefers_correct_same_root_order_and_handles_constant_labels():
    import torch
    y = torch.tensor([[1., 0.], [0., 0.]])
    good, _, good_rank = loss_function(torch.tensor([[2., -2.], [-2., -2.]]), y)
    bad, _, bad_rank = loss_function(torch.tensor([[-2., 2.], [-2., -2.]]), y)
    assert good < bad and good_rank < bad_rank
    loss, _, rank = loss_function(torch.zeros(2, 2, requires_grad=True), torch.zeros(2, 2))
    loss.backward()
    assert rank == 0 and torch.isfinite(loss)


def test_old_labels_require_exact_h8_and_root_identity():
    from scripts.collect_continuation_observability import verify
    natural = {'agentview_image': np.zeros((128, 128, 3), np.uint8),
               'robot0_eef_pos': np.zeros(3), 'robot0_gripper_qpos': np.zeros(2),
               'robot0_joint_pos': np.zeros(7), 'sim_state': np.zeros(4)}
    h8 = {'on': False, 'both_on': [False, False], 'grasp': False,
          'target': 'akita_black_bowl_1', 'm2': 0., 'both_m2': [0., 0.],
          'bowl_positions': np.zeros((2, 3)).tolist(), 'plate_position': [0., 0., 0.], 'reach': 0.}
    saved = {'root_proprio': np.zeros(12), 'root_rgb': natural['agentview_image'],
             'root_state': natural['sim_state'], 'h8_rgb': np.zeros((64, 128, 128, 3), np.uint8)}
    result = {'reference': natural, 'arrays': {'branch_rgb': np.zeros((4, 128, 128, 3), np.uint8)}, 'h8': h8.copy()}
    old = {'rows': [{'h8': h8} for _ in range(64)]}
    assert verify(result, old, saved, 0)['pass']
    result['arrays']['branch_rgb'][-1, 0, 0, 0] = 1
    with pytest.raises(ValueError, match='label reuse'):
        verify(result, old, saved, 0)
    result['arrays']['branch_rgb'][-1, 0, 0, 0] = 0
    result['h8']['m2'] = 1e-5
    with pytest.raises(ValueError, match='label reuse'):
        verify(result, old, saved, 0)


def test_arena_workspace_has_explicit_body_and_contact_binding():
    from types import SimpleNamespace
    from scripts.collect_continuation_observability import entity_bindings
    class Inner:
        def get_object(self, name):
            if name == 'main_table':
                return None
            return SimpleNamespace(root_body=name + '_body', contact_geoms=[name + '_collision'])
    bodies, geoms = entity_bindings(Inner())
    assert len(bodies) == len(geoms) == 8
    assert bodies[5] == 'table' and geoms[5] == ['table_collision']
