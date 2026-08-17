from __future__ import annotations

import json

import pytest
import torch

from test.fact_graph_analysis_corpus import CASES
from test.fact_graph_analysis_corpus import SNAPSHOT_PATH
from test.fact_graph_analysis_corpus import capture_corpus


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="the checked-in fact/config snapshot targets B200",
)
def test_fact_graph_analysis_corpus() -> None:
    expected = json.loads(SNAPSHOT_PATH.read_text())
    actual = capture_corpus()

    assert actual == expected
    assert {case.family for case in CASES if case.family != "vllm"} == {
        "rms_norm",
        "layer_norm",
        "cross_entropy",
        "jsd",
        "kl_div",
        "rms_norm_bwd",
        "layer_norm_bwd",
        "sum",
        "grpo_loss",
    }
    for case in CASES:
        record = actual[case.name]
        assert record["has_reduction"] is case.expects_reduction
        assert all(
            "error" not in heuristic for heuristic in record["heuristic_eligibility"]
        )

    vllm_reductions = [
        record
        for record in actual.values()
        if record["family"] == "vllm" and record["has_reduction"]
    ]
    assert vllm_reductions
    assert any(
        any(
            heuristic["name"].startswith("triton_reduction_")
            and heuristic.get("eligible") is True
            for heuristic in record["heuristic_eligibility"]
        )
        for record in vllm_reductions
    )
    assert any(record["reduction_heuristics"] for record in vllm_reductions)
