# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Throwaway diagnostic branch for #17377; never qualification evidence."""

import argparse
import copy
import json
import os
import subprocess
import threading
import time
from pathlib import Path


def prepare(kind: str, output: Path) -> tuple[Path, int]:
    profiling = os.environ.get("QWEN_RUNTIME_PROFILE", "1") == "1"
    source = Path("tests/dfx/perf/tests")
    if kind == "tts":
        configs = json.loads((source / "test_tts.json").read_text())[:3]
        for config in configs:
            params = copy.deepcopy(config["benchmark_params"][1])
            # Preserve the C8 workload and its corresponding baseline/policy.
            for name in ("num_prompts", "max_concurrency"):
                params[name] = params[name][:1]
            for metrics in params["baseline"].values():
                for metric, values in metrics.items():
                    metrics[metric] = values[:1] if isinstance(values, list) else values
            for policy in params.get("baseline_confirmation", {}).values():
                if isinstance(policy.get("tolerance"), list):
                    policy["tolerance"] = policy["tolerance"][:1]
            config["benchmark_params"] = [params]
    else:
        filename = f"test_qwen3_omni_{kind}_chunk.json"
        configs = json.loads((source / filename).read_text())
        index = 2 if kind == "async" else 3
        configs[0]["benchmark_params"] = [configs[0]["benchmark_params"][index]]

    for config in configs:
        profiler = {
            "profiler": "torch",
            "torch_profiler_dir": str(output / "profiles" / config["test_name"]),
            "torch_profiler_record_shapes": True,
            "torch_profiler_with_stack": False,
            "torch_profiler_use_gzip": True,
            "ignore_frontend": True,
            "max_iterations": 1000,
        }
        if profiling:
            overrides = config["server_params"].setdefault("stage_overrides", {})
            for stage in range(2 if kind == "tts" else 3):
                overrides.setdefault(str(stage), {})["profiler_config"] = profiler.copy()
        args = config["server_params"].setdefault("extra_cli_args", [])
        if kind != "tts" and "--stage-init-timeout" not in args:
            args.extend(["--stage-init-timeout", "900"])
        if profiling:
            config["benchmark_params"][0]["profile"] = True
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "diagnostic-config.json"
    destination.write_text(json.dumps(configs, indent=2) + "\n")
    return destination, len(configs)


def sample_resources(output: Path, stop: threading.Event) -> None:
    paths = [
        Path("/proc/loadavg"),
        Path("/proc/stat"),
        *Path("/sys/fs/cgroup").glob("cpu.*"),
        *Path("/sys/fs/cgroup").glob("cpuset.cpus*"),
        *Path("/sys/fs/cgroup").glob("memory.*"),
        *Path("/proc/pressure").glob("*"),
        *Path("/sys/class/drm").glob("card*/device/gpu_busy_percent"),
        *Path("/sys/class/drm").glob("card*/device/pp_dpm_sclk"),
        *Path("/sys/class/drm").glob("card*/device/mem_info_vram_used"),
    ]
    with (output / "resource-samples.jsonl").open("w") as stream:
        while not stop.is_set():
            values = {}
            for path in paths:
                try:
                    values[str(path)] = path.read_text()
                except OSError as exc:
                    values[str(path)] = str(exc)
            stream.write(json.dumps({"time_ns": time.time_ns(), "values": values}) + "\n")
            stream.flush()
            stop.wait(5)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("async", "no_async", "tts"))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    variant = os.environ.get("QWEN_RUNTIME_VARIANT", "profile")
    output = args.output or (
        Path(os.environ["BUILDKITE_BUILD_CHECKOUT_PATH"]) / "artifacts" / f"runtime-profile-{args.kind}-{variant}"
    )
    config, count = prepare(args.kind, output.resolve())
    (output / "experiment.json").write_text(
        json.dumps(
            {
                name: os.environ.get(name)
                for name in ("BUILDKITE_COMMIT", "QWEN_RUNTIME_PROFILE", "QWEN_RUNTIME_VARIANT", "MIOPEN_FIND_MODE")
            },
            indent=2,
        )
        + "\n"
    )
    if args.prepare_only:
        return 0
    env = os.environ | {
        "QWEN3_OMNI_ARTIFACT_DIR": str(output),
        "QWEN3_OMNI_CONFIG_FILE": str(config),
        "QWEN3_OMNI_EXPECTED_GPUS": "2",
        "QWEN3_OMNI_EXPECTED_RESULTS": str(count),
        "QWEN3_OMNI_EXPECTED_TESTS": str(count),
        "QWEN3_OMNI_REQUIRE_BASELINE": "1",
        "QWEN3_OMNI_MARK_EXPRESSION": "full_model and rocm and MI325 and cards_2",
        "QWEN3_OMNI_TEST_TARGET": "tests/dfx/perf/scripts/run_benchmark.py",
        "QWEN3_OMNI_TIMEOUT": "75m",
        "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
    }
    stop = threading.Event()
    monitor = threading.Thread(target=sample_resources, args=(output, stop), daemon=True)
    monitor.start()
    try:
        print(f"DIAGNOSTIC ONLY: variant={variant}; selected workloads, unchanged verdict rule.", flush=True)
        return subprocess.run(["bash", ".buildkite/amd/scripts/run-qwen3-omni-rocm-nightly.sh"], env=env).returncode
    finally:
        stop.set()
        monitor.join()


if __name__ == "__main__":
    raise SystemExit(main())
