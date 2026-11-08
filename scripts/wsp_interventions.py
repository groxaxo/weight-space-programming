"""Fixed forward-only interventions. No parameter updates, fitting or discovery.

Reimplements small intervention primitives against this project's exact BF16
contract; does not import a third-party trainer or intervention framework.
"""
from __future__ import annotations
import hashlib
import numpy as np
from direct_weights import bf16_float
from direct_causal import delta_stats
from pipeline import require

ARMS = ('base', 'noop', 'treatment', 'half', 'random_a', 'random_b', 'other_layer', 'other_position')
WIDTHS = {'input': 9216, 'output': 2560}


def validate(spec):
    require(type(spec) is dict and set(spec) == {'layer', 'location', 'position', 'off_target_position',
            'indices', 'operation', 'words'}, 'Unexpected fixed-intervention fields')
    require(type(spec['layer']) is int and 0 <= spec['layer'] < 32
            and spec['location'] in WIDTHS, 'Unsupported Qwen3.5-4B intervention site')
    require(type(spec['position']) is int and -1 <= spec['position'] < 512
            and type(spec['off_target_position']) is int and 0 <= spec['off_target_position'] < 512,
            'Invalid declared prompt position')
    indices = spec['indices']; width = WIDTHS[spec['location']]
    require(type(indices) is list and 1 <= len(indices) <= width
            and all(type(i) is int and 0 <= i < width for i in indices)
            and indices == sorted(set(indices)), 'Indices must be sorted unique activation coordinates')
    require(spec['operation'] in ('zero', 'replace', 'add'), 'Only fixed zero/replace/add interventions are supported')
    words = spec['words']
    require(type(words) is list and len(words) == (0 if spec['operation'] == 'zero' else len(indices))
            and all(type(w) is int and 0 <= w <= 65535 and w & 0x7f80 != 0x7f80 for w in words),
            'Expected finite explicit BF16 words, never a fitted function or executable artifact')


def bf16_words(values):
    """IEEE round-to-nearest-even for finite FP32, preserving BF16 word identity."""
    values = np.asarray(values, dtype='<f4')
    require(np.isfinite(values).all(), 'Nonfinite requested activation')
    raw = values.view('<u4')
    words = ((raw + np.uint32(0x7fff) + ((raw >> 16) & 1)) >> 16).astype('<u2')
    require(not np.any((words & 0x7f80) == 0x7f80), 'BF16 activation overflow; no clipping fallback')
    return words


def apply_words(before, spec, arm):
    """Return a private numeric copy and actual BF16 telemetry; never mutate input."""
    validate(spec); require(arm in ARMS, 'Unknown fixed control arm')
    before = np.asarray(before)
    require(before.dtype == np.dtype('<u2') and before.shape == (WIDTHS[spec['location']],)
            and not np.any((before & 0x7f80) == 0x7f80), 'Invalid finite BF16 activation vector')
    after = before.copy(); ix = np.asarray(spec['indices'], dtype=np.int64)
    source = bf16_float(before[ix])
    if arm not in ('base', 'noop'):
        if spec['operation'] == 'zero': target = np.zeros_like(source)
        elif spec['operation'] == 'replace': target = bf16_float(np.asarray(spec['words'], dtype='<u2'))
        else: target = source + bf16_float(np.asarray(spec['words'], dtype='<u2'))
        # First obtain the actual rounded full-treatment delta, then derive every
        # strength/random control from it. No score search or resampling.
        full = bf16_words(target)
        delta = bf16_float(full) - source
        if arm == 'half': delta *= np.float32(0.5)
        if arm in ('random_a', 'random_b'):
            signs = np.asarray([1 if hashlib.sha256(f'wsp-lab-v1:{arm}:{i}'.encode()).digest()[0] & 1 else -1
                                for i in spec['indices']], dtype=np.float32)
            delta *= signs
        after[ix] = (full if arm in ('treatment', 'other_layer', 'other_position')
                     else bf16_words(source + delta))
    stats = delta_stats(before, after)
    return after, {**stats, 'before_sha256': hashlib.sha256(before.tobytes()).hexdigest(),
                   'after_sha256': hashlib.sha256(after.tobytes()).hexdigest(),
                   'selected_entries': len(ix), 'arm': arm}


def norm_matched(actual, reference):
    """Match actual applied BF16 support and L2 within the frozen 5% tolerance."""
    for row in (actual, reference):
        require(type(row.get('changed_entries')) is int and row['changed_entries'] >= 0, 'Invalid changed count')
        require(type(row.get('delta_l2')) in (float, int) and np.isfinite(row['delta_l2']) and row['delta_l2'] >= 0,
                'Invalid intervention norm')
    return reference['delta_l2'] > 0 and all(abs(actual[k] - reference[k]) <= .05 * reference[k]
                                           for k in ('changed_entries', 'delta_l2'))


class FixedIntervention:
    """One prefill-only hook on a declared down_proj input/output. Cache-safe."""
    def __init__(self, spec, arm, prompt_tokens):
        validate(spec); require(arm in ARMS and type(prompt_tokens) is int and prompt_tokens > 0, 'Invalid hook plan')
        self.spec, self.arm, self.prompt_tokens = spec, arm, prompt_tokens
        self.layer = (spec['layer'] + 1) % 32 if arm == 'other_layer' else spec['layer']
        selected = spec['off_target_position'] if arm == 'other_position' else spec['position']
        self.position = prompt_tokens - 1 if selected == -1 else selected
        original = prompt_tokens - 1 if spec['position'] == -1 else spec['position']
        require(0 <= self.position < prompt_tokens and 0 <= original < prompt_tokens
                and spec['off_target_position'] < prompt_tokens and spec['off_target_position'] != original,
                'Off-target and target must be distinct actual prompt positions')
        self.telemetry = None; self.before = self.after = None; self.handle = None

    def _apply(self, value):
        import torch
        if self.telemetry is not None: return value
        require(not torch.is_grad_enabled() and not value.requires_grad, 'Interventions require inference mode')
        require(value.dtype == torch.bfloat16 and value.ndim == 3
                and tuple(value.shape[:2]) == (1, self.prompt_tokens)
                and value.shape[2] == WIDTHS[self.spec['location']], 'Unexpected prefill activation shape/dtype')
        self.before = value[0, self.position].detach().contiguous().view(torch.uint16).numpy().copy()
        self.after, self.telemetry = apply_words(self.before, self.spec, self.arm)
        self.telemetry.update(layer=self.layer, location=self.spec['location'], position=self.position)
        if self.arm in ('base', 'noop'): return value
        result = value.clone()
        result[0, self.position] = torch.from_numpy(self.after.copy()).view(torch.bfloat16)
        return result

    def install(self, model):
        require(self.handle is None, 'Hook already installed')
        module = model.model.layers[self.layer].mlp.down_proj
        if self.spec['location'] == 'input':
            self.handle = module.register_forward_pre_hook(lambda _m, args: (self._apply(args[0]), *args[1:]))
        else:
            self.handle = module.register_forward_hook(lambda _m, _args, output: self._apply(output))
        return self

    def close(self):
        if self.handle is not None: self.handle.remove(); self.handle = None
