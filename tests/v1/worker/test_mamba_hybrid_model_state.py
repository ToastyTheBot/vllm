# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backends.recoverssm_metadata import (
    RecoverSSMMetadata,
    RecoverSSMPostprocessMetadata,
)
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
from vllm.v1.worker.gpu.model_states.recoverssm import RecoverSSMState


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
@pytest.mark.parametrize(("num_sampled", "expected_value"), [(0, 1), (3, 3)])
def test_postprocess_state_scalar_with_int32_mapping(
    num_sampled: int, expected_value: int
) -> None:
    state = object.__new__(MambaHybridModelState)
    state.num_accepted_tokens_gpu = torch.full(
        (4,), 9, dtype=torch.int32, device="cuda"
    )
    state._align_mode = False
    state.recoverssm = None
    state._mamba_ctx = None
    idx_mapping = torch.tensor([2, -1, 0], dtype=torch.int32, device="cuda")

    state.postprocess_state(idx_mapping, num_sampled)

    expected = torch.tensor(
        [expected_value, 9, expected_value, 9], dtype=torch.int32, device="cuda"
    )
    torch.testing.assert_close(state.num_accepted_tokens_gpu, expected)


def test_recoverssm_commits_accepted_window_after_v2_sampling() -> None:
    state = RecoverSSMState()
    metadata = Mock(spec=RecoverSSMMetadata)
    metadata.commit_recoverssm_state.return_value = None
    num_sampled = torch.tensor([3, 1], dtype=torch.int32)
    idx_mapping = torch.tensor([0, 1], dtype=torch.int32)
    num_accepted_tokens = torch.ones(2, dtype=torch.int32)
    group = SimpleNamespace(layer_names=["layer"])

    state.record_step({"layer": metadata}, [[group]], for_capture=False)
    state.commit_step(
        num_sampled,
        idx_mapping,
        state_indices=None,
        num_accepted_tokens=num_accepted_tokens,
    )
    state.commit_step(
        num_sampled,
        idx_mapping,
        state_indices=None,
        num_accepted_tokens=num_accepted_tokens,
    )

    metadata.commit_recoverssm_state.assert_called_once_with(num_sampled)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_recoverssm_align_tracks_mixed_batch_state_and_neutralizes_copy_bias() -> None:
    state = object.__new__(MambaHybridModelState)
    state._align_mode = True
    state._mamba_ctx = None
    state._mamba_state_idx_gpu = torch.full((5,), -1, dtype=torch.int32, device="cuda")
    state.recoverssm = RecoverSSMState()
    state.num_accepted_tokens_gpu = torch.full(
        (5,), 9, dtype=torch.int32, device="cuda"
    )
    metadata = Mock(spec=RecoverSSMMetadata)
    metadata.commit_recoverssm_state.return_value = RecoverSSMPostprocessMetadata(
        num_spec_decodes=1,
        request_indices=torch.tensor([1], dtype=torch.int32, device="cuda"),
        num_computed_tokens=torch.tensor([6, 7], dtype=torch.int32, device="cuda"),
        block_size=8,
        block_table=torch.zeros((2, 4), dtype=torch.int32, device="cuda"),
    )
    num_sampled = torch.tensor([2, 3], dtype=torch.int32, device="cuda")
    idx_mapping = torch.tensor([3, 1], dtype=torch.int32, device="cuda")
    group = SimpleNamespace(layer_names=["layer"])

    state.recoverssm.record_step({"layer": metadata}, [[group]], for_capture=False)

    state.postprocess_state(idx_mapping, num_sampled)

    expected_state_indices = [-1, 1, -1, -1, -1]
    assert state._mamba_state_idx_gpu.tolist() == expected_state_indices
    expected_accepted = [9, 1, 9, 2, 9]
    assert state.num_accepted_tokens_gpu.tolist() == expected_accepted


# --------------------------------------------------------------------------- #
# Regression tests for the align-mode state-column seeding.
#
# MambaHybridModelState.add_request used to seed the running state column with
# cache_config.block_size. EngineCore rewrites that field to
# min(g.kv_cache_spec.block_size for g in kv_cache_groups) once the KV cache
# config is known, so on a hybrid model it becomes the SMALLEST group's block
# size, not the mamba group's. On Qwen3.8-Flash-Next that is the PLE
# short-conv group's 4 against a mamba block size of 1568 -- a 392x overshoot
# that walks off the mamba block table, reads a garbage block id, and faults in
# precopy_mamba_align_fused_kernel.
#
# num_computed_tokens == 0 yields -1 for any positive divisor, which is why
# only prefix-cache hits ever crashed -- so the seeding tests below use a
# non-zero num_computed_tokens on purpose.
# --------------------------------------------------------------------------- #

# Realistic values from the failure: mamba block 1568, and a cache_config
# block_size of 4 contributed by a co-resident short-conv group.
_MAMBA_BLOCK_SIZE = 1568
_SMALLEST_GROUP_BLOCK_SIZE = 4


def _align_state(*, mamba_spec_block_size, mamba_block_size, block_size):
    """A bare align-mode state with only what the seeding path touches."""
    state = object.__new__(MambaHybridModelState)
    state._align_mode = True
    state.cache_config = SimpleNamespace(
        block_size=block_size, mamba_block_size=mamba_block_size
    )
    state._mamba_spec = (
        None
        if mamba_spec_block_size is None
        else SimpleNamespace(block_size=mamba_spec_block_size)
    )
    return state


def test_align_block_size_prefers_the_mamba_spec() -> None:
    state = _align_state(
        mamba_spec_block_size=_MAMBA_BLOCK_SIZE,
        mamba_block_size=_MAMBA_BLOCK_SIZE,
        block_size=_SMALLEST_GROUP_BLOCK_SIZE,
    )
    assert state._mamba_align_block_size() == _MAMBA_BLOCK_SIZE


def test_align_block_size_falls_back_to_mamba_block_size() -> None:
    """add_request can run before the first preprocess_state resolves the spec."""
    state = _align_state(
        mamba_spec_block_size=None,
        mamba_block_size=_MAMBA_BLOCK_SIZE,
        block_size=_SMALLEST_GROUP_BLOCK_SIZE,
    )
    assert state._mamba_align_block_size() == _MAMBA_BLOCK_SIZE


def test_align_block_size_never_falls_back_to_cache_block_size() -> None:
    """With no mamba block size available it must raise, not silently use the
    smallest group's block size -- that fallback is the original defect."""
    state = _align_state(
        mamba_spec_block_size=None,
        mamba_block_size=None,
        block_size=_SMALLEST_GROUP_BLOCK_SIZE,
    )
    with pytest.raises(RuntimeError, match="mamba block size"):
        state._mamba_align_block_size()


@pytest.mark.parametrize("num_computed_tokens", [10975, 45472, 136510])
def test_add_request_seeds_state_column_with_the_mamba_block_size(
    monkeypatch: pytest.MonkeyPatch, num_computed_tokens: int
) -> None:
    """The regression itself: a prefix-cache hit must seed from the mamba block
    size, not from the (much smaller) rewritten cache_config.block_size."""
    state = _align_state(
        mamba_spec_block_size=_MAMBA_BLOCK_SIZE,
        mamba_block_size=_MAMBA_BLOCK_SIZE,
        block_size=_SMALLEST_GROUP_BLOCK_SIZE,
    )
    state.num_accepted_tokens_gpu = torch.zeros(4, dtype=torch.int32)
    state._mamba_state_idx_gpu = torch.zeros(4, dtype=torch.int32)
    monkeypatch.setattr(
        MambaHybridModelState.__bases__[0], "add_request", lambda *a, **k: None
    )

    state.add_request(2, Mock(num_computed_tokens=num_computed_tokens))

    expected = (num_computed_tokens - 1) // _MAMBA_BLOCK_SIZE
    buggy = (num_computed_tokens - 1) // _SMALLEST_GROUP_BLOCK_SIZE
    assert int(state._mamba_state_idx_gpu[2]) == expected
    # Guard the specific wrong value, so a future "simplification" back to
    # cache_config.block_size fails loudly here rather than as a CUDA fault.
    assert int(state._mamba_state_idx_gpu[2]) != buggy


def test_add_request_seeds_minus_one_for_a_fresh_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """num_computed_tokens == 0 must stay -1 (nothing to copy) -- the property
    that kept non-prefix-cache traffic working even with the bug present."""
    state = _align_state(
        mamba_spec_block_size=_MAMBA_BLOCK_SIZE,
        mamba_block_size=_MAMBA_BLOCK_SIZE,
        block_size=_SMALLEST_GROUP_BLOCK_SIZE,
    )
    state.num_accepted_tokens_gpu = torch.zeros(4, dtype=torch.int32)
    state._mamba_state_idx_gpu = torch.zeros(4, dtype=torch.int32)
    monkeypatch.setattr(
        MambaHybridModelState.__bases__[0], "add_request", lambda *a, **k: None
    )

    state.add_request(1, Mock(num_computed_tokens=0))

    assert int(state._mamba_state_idx_gpu[1]) == -1
