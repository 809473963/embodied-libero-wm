"""Causal operational relation evidence. No simulator, outcome or policy imports."""
import numpy as np

EVENTS = ('bilateral_attachment', 'detached_attachment', 'lift_hold_evidence',
          'released_on_quiet', 'release_after_attachment')


def labels(a, cfg, entity=0):
    """int8 {-1 unknown, 0 predicate false, 1 predicate true}, indexed at window END.

    actions[t] transitions state[t] to state[t+1]. These labels describe observed
    evidence, never certify unmeasured load-bearing, clearance or future stability.
    """
    pose = np.asarray(a['pose']); n = len(pose); w = cfg['window_intervals']
    assert entity in (0, 1) and pose.shape[1:] == (8, 13)
    assert np.array_equal(a['step'], np.arange(n)), 'non-contiguous recording'
    for key in ('pose', 'grip_pos', 'grip_rot'):
        assert np.isfinite(a[key]).all(), key
    pos = pose[:, entity, :3]
    local = np.einsum('tji,tj->ti', a['grip_rot'], pos-a['grip_pos'])
    bilateral = a['pad_contact'][:, :, entity].all(1)
    grip = a['gripper_contact'][:, entity]
    external = a['non_gripper_contact'][:, entity]
    on = a['on'][:, entity]
    speed = np.linalg.norm(pose[:, entity, 7:10], axis=1)
    angular = np.linalg.norm(pose[:, entity, 10:13], axis=1)
    values = np.full((n, len(EVENTS)), -1, np.int8)
    reference = np.full(n, -1, np.int32)
    # Inputs are continuous object relations, not episode outcome bits.
    pair_position = pose[:, :, None, :3] - pose[:, None, :, :3]
    pair_velocity = pose[:, :, None, 7:10] - pose[:, None, :, 7:10]
    for t in range(w, n):
        start = t-w; sl = slice(start, t+1)
        attached = bool(bilateral[sl].all() and
                        np.linalg.norm(local[sl]-local[start], axis=1).max() <= cfg['local_drift_m'])
        detached = attached and not external[sl].any()
        values[t, 0:2] = attached, detached
        prior = np.flatnonzero(external[:start])
        if len(prior):
            reference[t] = prior[-1]
        if not detached:
            values[t, 2] = 0
        elif len(prior):
            values[t, 2] = bool(pos[sl, 2].min()-pos[prior[-1], 2] > cfg['rise_m'])
        quiet = bool(on[sl].all() and not grip[sl].any() and external[sl].all()
                     and (speed[sl] < cfg['linear_speed_m_s']).all()
                     and (angular[sl] < cfg['angular_speed_rad_s']).all()
                     and np.linalg.norm(pos[sl]-pos[start], axis=1).max() <= cfg['placement_drift_m'])
        values[t, 3] = quiet
        # Require real previously observed detached attachment, not just a touch.
        # History can certify occurrence; absence before the recording is unknown.
        if not quiet:
            values[t, 4] = 0
        elif (values[:start, 1] == 1).any():
            values[t, 4] = 1
    onset = np.full_like(values, -1)
    for t in range(1, n):
        valid = (values[t-1] >= 0) & (values[t] >= 0)
        onset[t, valid] = ((values[t-1, valid] == 0) & (values[t, valid] == 1))
    return dict(values=values, onset=onset, lift_reference_step=reference,
                local_position=local, pair_position=pair_position, pair_velocity=pair_velocity)


def future_targets(values, horizons):
    """Future endpoint labels, with explicit censoring; never a model input."""
    out = np.full((len(values), len(horizons), values.shape[1]), -1, np.int8)
    for j, h in enumerate(horizons):
        assert h > 0
        if h < len(values):
            out[:-h, j] = values[h:]
    return out
