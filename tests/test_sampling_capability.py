# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Host-only tests for model-specific device-sampling policy."""

from types import SimpleNamespace

import pytest
import torch

from vllm_tt_plugin.model_input import TTSamplingParams
from vllm_tt_plugin.model_runner import TTModelRunner


def _params(*, presence=0.0, frequency=0.0, repetition=1.0):
    return TTSamplingParams(
        temperature=torch.tensor([0.1]),
        top_k=torch.tensor([32]),
        top_p=torch.tensor([0.95]),
        presence_penalty=torch.tensor([presence]),
        frequency_penalty=torch.tensor([frequency]),
        repetition_penalty=torch.tensor([repetition]),
        seed=torch.tensor([1234]),
    )


def _runner(callback):
    return SimpleNamespace(
        sample_on_device_mode="all",
        num_devices=4,
        tt_data_parallel_size=1,
        input_batch=SimpleNamespace(
            no_allowed_token_ids=True,
            sampling=SimpleNamespace(
                bad_words_token_ids={}, has_active_logitsprocs=lambda: False
            ),
            max_num_logprobs=None,
        ),
        model_config=SimpleNamespace(logits_processors=[]),
        supports_topk_logprobs=False,
        _supports_device_sampling=callback,
    )


@pytest.mark.parametrize(
    "params",
    [
        _params(presence=1.2),
        _params(frequency=1.2),
        _params(repetition=1.5),
    ],
)
def test_model_policy_can_route_penalties_to_host(params):
    seen = []

    def supports_device_sampling(actual, *, is_decode):
        seen.append((actual, is_decode))
        return False

    runner = _runner(supports_device_sampling)
    assert not TTModelRunner.check_perform_device_sampling(
        runner,
        is_decode=True,
        has_structured_outputs=False,
        sampling_params=params,
    )
    assert seen == [(params, True)]


def test_legacy_model_without_policy_keeps_device_sampling():
    runner = _runner(None)
    assert TTModelRunner.check_perform_device_sampling(
        runner,
        is_decode=True,
        has_structured_outputs=False,
        sampling_params=_params(),
    )


def test_model_policy_error_is_not_hidden():
    def reject(*_args, **_kwargs):
        raise ValueError("explicit compatibility required")

    runner = _runner(reject)
    with pytest.raises(ValueError, match="explicit compatibility required"):
        TTModelRunner.check_perform_device_sampling(
            runner,
            is_decode=False,
            has_structured_outputs=False,
            sampling_params=_params(presence=1.2),
        )


def test_shared_host_only_gate_precedes_model_policy():
    calls = []
    runner = _runner(lambda *_args, **_kwargs: calls.append(True) or True)
    assert not TTModelRunner.check_perform_device_sampling(
        runner,
        is_decode=True,
        has_structured_outputs=True,
        sampling_params=_params(),
    )
    assert calls == []
