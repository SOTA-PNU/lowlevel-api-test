import json
import subprocess
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("rebel")

from triton_tests import common
from triton_tests.tests import npu_language as npu


def test_common_tolerance_stays_strict_and_npu_can_override():
    actual = torch.tensor([1.005])
    expected = torch.tensor([1.0])

    assert not common._compare_tensors(actual, expected)[0]
    assert common._compare_tensors(
        actual, expected, rtol=2e-2, atol=2e-2
    )[0]


@pytest.mark.parametrize(
    "raw",
    [
        "{",
        "[]",
        '{"ok": true}',
        '{"ok": 1, "has_reference": true, "max_abs": 0, "max_rel": 0}',
        '{"ok": true, "has_reference": true, "max_abs": NaN, "max_rel": 0}',
        '{"ok": true, "has_reference": true, "max_abs": 0, "max_rel": 0, "dtype": null}',
        (
            '{"ok": true, "has_reference": true, "max_abs": '
            + "1" * 4000
            + ', "max_rel": 0}'
        ),
    ],
)
def test_worker_payload_rejects_malformed_values(raw):
    with pytest.raises(ValueError):
        npu._decode_worker_payload(raw)


def test_power_stability_requires_three_observations():
    assert not npu._power_is_stable([70.0, 70.0])
    assert npu._power_is_stable([70.0, 71.0, 70.5])
    assert not npu._power_is_stable([70.0, 80.0, 70.0])


def test_npu_tolerance_preserves_dot_exception():
    assert npu._npu_tolerance("dot") == pytest.approx(0.2)
    assert npu._npu_tolerance("add") == pytest.approx(0.02)


def test_shared_card_skips_energy(monkeypatch):
    def shared_snapshot(*args, **kwargs):
        raise npu._SharedCardError("shared-card")

    monkeypatch.setattr(npu, "_target_power_snapshot", shared_snapshot)

    assert npu._measure_energy_mj_per_call(
        lambda *args: None, (), 3.0
    ) == (None, None, "shared-card")


@pytest.mark.parametrize(
    ("first_response", "expected_detail"),
    [
        (subprocess.TimeoutExpired(["worker"], 320), "timed out"),
        (
            subprocess.CompletedProcess(
                ["worker"], 0,
                stdout="RBLN_OP_DTYPE=fp32\nRBLN_OP_RESULT={\n",
                stderr="",
            ),
            "invalid RBLN worker payload",
        ),
    ],
)

def test_bad_worker_is_isolated(monkeypatch, first_response, expected_detail):
    valid_payload = {
        "ok": True,
        "has_reference": True,
        "max_abs": 0.0,
        "max_rel": 0.0,
        "dtype": "fp32",
        "ms": 0.1,
        "timer_source": "mock",
        "timer_warning": None,
        "energy_mj_per_call": None,
        "energy_source": None,
        "energy_warning": None,
    }
    responses = iter(
        [
            first_response,
            subprocess.CompletedProcess(
                ["worker"],
                0,
                stdout=(
                    "RBLN_OP_DTYPE=fp32\nRBLN_OP_RESULT="
                    + json.dumps(valid_payload)
                    + "\n"
                ),
                stderr="",
            ),
        ]
    )

    def fake_run(*args, **kwargs):
        response = next(responses)
        if isinstance(response, BaseException):
            raise response
        return response

    monkeypatch.setattr(npu.subprocess, "run", fake_run)
    monkeypatch.setattr(npu, "_selected_ops", lambda only: ["add", "exp"])
    monkeypatch.setattr(
        npu, "positive_input", lambda device="cpu": torch.ones(1)
    )
    monkeypatch.setattr(common, "RUNTIME_DEVICE", "npu")
    monkeypatch.setattr(common, "RUNTIME_DEVICE_LABEL", "NPU mock")

    results = npu.run(
        SimpleNamespace(only="", warmup=0, rep=1, energy_seconds=0.0)
    )

    assert results["tl.add"].result == common.TestResult.ERROR
    assert expected_detail in results["tl.add"].detail
    assert results["tl.exp"].result == common.TestResult.PASS
