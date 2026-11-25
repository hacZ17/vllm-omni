# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU contracts for the ROCm DiT accuracy lane; no model imports or GPU runs."""

import ast
import sys
from collections.abc import Callable
from pathlib import Path
from shlex import split
from types import ModuleType
from typing import cast

import pytest
import yaml

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_ACCURACY_FILE = Path("tests/e2e/accuracy/test_hunyuan_image3.py")
_NODE = f"{_ACCURACY_FILE.as_posix()}::test_hunyuan_image3_dit_tp4_it2i_online"


def test_hunyuan_dit_accuracy_has_one_guarded_nightly_owner() -> None:
    steps = []
    for tier in ("ready", "merge", "nightly", "weekly"):
        pipeline = yaml.safe_load(Path(f".buildkite/amd/test-amd-{tier}.yml").read_text(encoding="utf-8"))
        for group in pipeline["steps"]:
            for step in group.get("steps", [group]):
                if _NODE in "\n".join(step.get("commands", [])):
                    steps.append((tier, step))
    assert len(steps) == 1
    tier, step = steps[0]
    assert tier == "nightly"
    assert step["agent_pool"] == "mi300_4"
    assert step["depends_on"] == "amd-build"
    assert step["mirror_hardwares"] == ["amdproduction"]
    assert step["grade"] == "Blocking"
    assert "retry" not in step
    assert step["timeout_in_minutes"] == 90
    assert step["source_file_dependencies"] == ["diffusion_hunyuan_image3_accuracy", "rocm_diffusion_nightly_runner"]
    assert len(step["commands"]) == 1
    assert split(step["commands"][0]) == [
        "bash",
        ".buildkite/amd/scripts/run-rocm-diffusion-nightly.sh",
        "4",
        "1",
        "75m",
        "rocm-hunyuan-image3-dit-tp4",
        _NODE,
        "-m",
        "full_model and diffusion and rocm and MI325 and cards_4",
    ]
    assert step["env"]["HUNYUAN_IMAGE3_MODEL"] == "tencent/HunyuanImage-3.0-Instruct-Distil"
    assert step["env"]["DIFFUSION_ATTENTION_BACKEND"] == "TORCH_SDPA"
    assert step["env"]["ACC_BENCH_RESULT_DIR"] == (
        "$${BUILDKITE_BUILD_CHECKOUT_PATH:?}/artifacts/rocm-hunyuan-image3-dit-tp4/accuracy-results"
    )
    assert step["artifact_paths"] == [
        "artifacts/rocm-hunyuan-image3-dit-tp4/**/*",
        "tests/e2e/accuracy/artifacts/**/*",
    ]


@pytest.mark.parametrize("platform", ["cuda", "rocm", "npu"])
def test_hunyuan_dit_rocm_override_preserves_tp4_and_reference(monkeypatch: pytest.MonkeyPatch, platform: str) -> None:
    """Exercise the real selectors without importing the GPU/model dependency graph."""

    class PlatformStub:
        def is_rocm(self) -> bool:
            return platform == "rocm"

        def is_npu(self) -> bool:
            return platform == "npu"

    module = ModuleType("vllm_omni.platforms")
    module.__dict__["current_omni_platform"] = PlatformStub()
    monkeypatch.setitem(sys.modules, "vllm_omni.platforms", module)
    tree = ast.parse(_ACCURACY_FILE.read_text(encoding="utf-8"))
    selectors = ast.Module(
        body=[
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in {"_dit_deploy_config", "_dit_it2i_reference_path"}
        ],
        type_ignores=[],
    )
    base_path = "vllm_omni/deploy/hunyuan_image3_dit.yaml"
    modify_calls: list[tuple[str, dict]] = []

    def modify(config: str, updates: dict) -> str:
        modify_calls.append((config, updates))
        return "rocm-dit.yaml"

    namespace = {
        "Path": Path,
        "get_deploy_config_path": lambda name: f"vllm_omni/deploy/{name}",
        "modify_stage_config": modify,
        "HUNYUAN_DIT_IMAGE_REF_PATH": Path("cuda-reference.png"),
        "HUNYUAN_DIT_IMAGE_REF_PATH_NPU": Path("npu-reference.png"),
    }
    exec(compile(selectors, str(_ACCURACY_FILE), "exec"), namespace)
    config = cast(Callable[[], str], namespace["_dit_deploy_config"])()
    if platform == "rocm":
        assert config == "rocm-dit.yaml"
        assert modify_calls == [(base_path, {"stages": {0: {"parallel_config.enable_expert_parallel": False}}})]
    else:
        assert config == base_path
        assert modify_calls == []
    base = yaml.safe_load(Path(base_path).read_text(encoding="utf-8"))
    assert base["stages"][0]["parallel_config"]["tensor_parallel_size"] == 4
    assert base["stages"][0]["parallel_config"]["enable_expert_parallel"] is True
    assert cast(Callable[[], Path], namespace["_dit_it2i_reference_path"])() == Path(
        "npu-reference.png" if platform == "npu" else "cuda-reference.png"
    )


def test_hunyuan_dit_only_enables_the_existing_four_card_accuracy_case() -> None:
    tree = ast.parse(_ACCURACY_FILE.read_text(encoding="utf-8"))
    rocm_cases = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call) or not isinstance(decorator.func, ast.Name):
                continue
            if decorator.func.id != "hardware_test":
                continue
            arguments = {keyword.arg: ast.literal_eval(keyword.value) for keyword in decorator.keywords}
            if "rocm" in arguments["res"]:
                rocm_cases.append(node.name)
                assert arguments["res"]["rocm"] == "MI325"
                assert arguments["num_cards"] == 4
    assert rocm_cases == [_NODE.split("::")[1]]
