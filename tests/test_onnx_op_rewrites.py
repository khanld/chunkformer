#!/usr/bin/env python3
"""Exactness tests for the ONNX-safe op rewrites used during export.

Two native ops are not representable by the (legacy) ONNX exporter and are
transparently rewritten when ``torch.onnx.is_in_onnx_export()`` is true:

* ``Tensor.unfold(dim, size, step)`` -> ``chunkformer.utils.mask.onnx_unfold``
  (gather based), used by limited-context attention and dynamic convolution.
* relative-attention ``rel_shift`` ``as_strided`` -> a ``torch.gather`` based
  shift in ``ChunkAttentionWithRelativeRightContext.rel_shift``.

These tests assert the rewritten ops produce *bit-identical* results to the
native ops across the shapes/dtypes/contexts actually exercised by the model.
"""

from unittest import mock

import pytest
import torch

from chunkformer.modules.attention import ChunkAttentionWithRelativeRightContext
from chunkformer.utils.mask import onnx_unfold

# ---------------------------------------------------------------------------
# Tensor.unfold  ->  onnx_unfold
# ---------------------------------------------------------------------------

# (shape, dim, size, step) covering every call site in the model:
#   attention: q  unfold(dim=1, chunk, chunk)
#   attention: kv unfold(dim=2, l+c+r, chunk)
#   attention: mask unfold(dim=-1, chunk, chunk) and (dim=-1, l+c+r, chunk)
#   conv:      unfold(dim=-1, lorder+chunk, chunk)
_UNFOLD_CASES = [
    # (shape, dim, size, step, note)
    ((2, 12, 4, 8), 1, 4, 4, "attn-q exact"),
    ((2, 13, 4, 8), 1, 4, 4, "attn-q (needs trailing window dropped)"),
    ((1, 4, 12, 16), 2, 6, 4, "attn-kv overlapping l+c+r"),
    ((3, 1, 20), -1, 5, 5, "mask-q non-overlap"),
    ((3, 1, 20), -1, 9, 5, "mask-kv overlap"),
    ((2, 32, 24), -1, 9, 8, "conv lorder+chunk"),
    ((1, 16, 8), -1, 8, 8, "single window (n=1)"),
    ((1, 16, 9), 2, 9, 9, "single window exact fit"),
    ((4, 7, 5, 40), -1, 7, 3, "4d last-dim overlap"),
    ((2, 3, 4, 50), 3, 11, 7, "4d explicit positive dim"),
]


@pytest.mark.parametrize("shape,dim,size,step,note", _UNFOLD_CASES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bool, torch.int64])
def test_onnx_unfold_matches_native(shape, dim, size, step, note, dtype):
    if dtype == torch.bool:
        x = torch.rand(shape) > 0.5
    elif dtype == torch.int64:
        x = torch.randint(-50, 50, shape)
    else:
        x = torch.randn(shape)

    native = x.unfold(dim, size, step)
    rewritten = onnx_unfold(x, dim, size, step)

    assert rewritten.shape == native.shape, f"{note}: shape {rewritten.shape} != {native.shape}"
    assert rewritten.dtype == native.dtype, note
    assert torch.equal(rewritten, native), f"{note}: values differ"


def test_onnx_unfold_keeps_window_count_dynamic_symbolically():
    """The number of windows must come from the data, not be baked as a constant
    (this is what lets a single exported graph handle variable-length audio)."""
    # Same (size, step) but different lengths -> different window counts, both OK.
    for length, expected_n in [(20, 4), (40, 9), (8, 1)]:
        x = torch.randn(2, 3, length)
        out = onnx_unfold(x, -1, 8, 4)
        assert out.shape == x.unfold(-1, 8, 4).shape
        assert out.shape[2] == expected_n


# ---------------------------------------------------------------------------
# rel_shift:  as_strided  ->  gather
# ---------------------------------------------------------------------------

# (batch, head, time1, left_context, right_context). The relative-pos tensor
# width must be n = 2*time1 - 1 + left + right (as produced by the model).
_REL_SHIFT_CASES = [
    (1, 4, 1, 0, 0),
    (2, 8, 5, 0, 0),
    (2, 4, 7, 3, 2),
    (1, 2, 10, 6, 0),
    (3, 4, 6, 0, 4),
    (2, 8, 16, 8, 8),
    (1, 1, 25, 12, 5),
]


def _make_attn(num_heads):
    torch.manual_seed(0)
    # n_feat must be divisible by num_heads; value is irrelevant for rel_shift.
    return ChunkAttentionWithRelativeRightContext(num_heads, num_heads * 4, 0.0)


@pytest.mark.parametrize("batch,head,time1,left,right", _REL_SHIFT_CASES)
def test_rel_shift_gather_matches_as_strided(batch, head, time1, left, right):
    attn = _make_attn(head)
    n = 2 * time1 - 1 + left + right
    x = torch.randn(batch, head, time1, n)

    # Native (as_strided) branch.
    native = attn.rel_shift(x, left, right)

    # Force the ONNX/gather branch.
    with mock.patch("torch.onnx.is_in_onnx_export", return_value=True):
        rewritten = attn.rel_shift(x, left, right)

    expected_time2 = time1 + left + right
    assert native.shape == (batch, head, time1, expected_time2)
    assert rewritten.shape == native.shape
    assert torch.equal(rewritten, native), "gather rel_shift differs from as_strided"


def test_rel_shift_formula_explicit():
    """Spot-check the documented identity out[..., i, j] = x[..., i, (time1-1)-i+j]
    on a tiny tensor with known values."""
    attn = _make_attn(1)
    time1, left, right = 3, 0, 0
    n = 2 * time1 - 1 + left + right  # 5
    x = torch.arange(time1 * n, dtype=torch.float32).view(1, 1, time1, n)

    with mock.patch("torch.onnx.is_in_onnx_export", return_value=True):
        out = attn.rel_shift(x, left, right)

    time2 = time1 + left + right
    ref = torch.empty(1, 1, time1, time2)
    for i in range(time1):
        for j in range(time2):
            ref[0, 0, i, j] = x[0, 0, i, (time1 - 1) - i + j]
    assert torch.equal(out, ref)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
