import ast
from pathlib import Path
import torch

from triton_tests import common


def test_compare_tensors_uses_1e_2_for_rtol_and_atol():
    zero = torch.tensor([0.0], dtype=torch.float64)
    ten = torch.tensor([10.0], dtype=torch.float64)

    assert common._compare_tensors(
        torch.tensor([0.9e-2], dtype=torch.float64), zero
    )[0]
    assert not common._compare_tensors(
        torch.tensor([1.1e-2], dtype=torch.float64), zero
    )[0]
    assert common._compare_tensors(
        torch.tensor([10.109], dtype=torch.float64), ten
    )[0]
    assert not common._compare_tensors(
        torch.tensor([10.111], dtype=torch.float64), ten
    )[0]


def test_operator_suites_do_not_override_common_tolerance():
    tests_dir = Path(common.REPO_ROOT) / "triton_tests" / "tests"
    callers = set()
    overrides = []

    for path in sorted(tests_dir.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name) and node.func.id == "_compare_tensors":
                callers.add(path.name)
                keywords = {keyword.arg for keyword in node.keywords}
                if len(node.args) > 2 or keywords & {"rtol", "atol"}:
                    overrides.append(f"{path.name}:{node.lineno}")

    assert {
        "cuda_libdevice.py",
        "npu_language.py",
        "triton_language.py",
    } <= callers
    assert overrides == []
