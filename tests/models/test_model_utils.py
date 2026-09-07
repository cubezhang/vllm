# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.models.utils import maybe_fuse_shared_experts


def test_maybe_fuse_shared_experts_scalar_scale() -> None:
    scale = torch.tensor(0.5)
    weights = [("model.layers.0.mlp.shared_mlp.gate_proj.weight_scale", scale)]

    fused_weights = list(
        maybe_fuse_shared_experts(
            weights,
            n_routed_experts=192,
            n_shared_experts=1,
            ckpt_prefix="mlp.shared_mlp",
            enabled=True,
        )
    )

    assert fused_weights == [
        ("model.layers.0.mlp.experts.192.gate_proj.weight_scale", scale)
    ]


def test_maybe_fuse_shared_experts_rejects_scalar_split() -> None:
    weights = [
        (
            "model.layers.0.mlp.shared_experts.gate_proj.weight_scale",
            torch.tensor(0.5),
        )
    ]

    with pytest.raises(ValueError, match="cannot be split"):
        list(
            maybe_fuse_shared_experts(
                weights,
                n_routed_experts=192,
                n_shared_experts=2,
                enabled=True,
            )
        )
