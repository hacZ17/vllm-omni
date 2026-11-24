# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Diagnostic only: appended inside the pod to serving_chat.py, never merged.
import cProfile as _diag_cprofile  # noqa: N813
import functools as _diag_functools
import json as _diag_json
import os as _diag_os
import time as _diag_time
import uuid as _diag_uuid
from pathlib import Path as _DiagPath

from vllm.multimodal.processing import BaseMultiModalProcessor as _DiagProcessor

_diag_dir = _DiagPath(_diag_os.environ["QWEN_FRONTEND_PROFILE_DIR"])
_diag_dir.mkdir(parents=True, exist_ok=True)
_diag_apply_original = _DiagProcessor.apply
_diag_preprocess_original = OmniOpenAIServingChat._preprocess_chat


@_diag_functools.wraps(_diag_apply_original)
def _diag_apply(self, *args, **kwargs):
    profiler = _diag_cprofile.Profile()
    started = _diag_time.time_ns()
    try:
        return profiler.runcall(_diag_apply_original, self, *args, **kwargs)
    finally:
        profiler.create_stats()
        rows = [
            dict(
                file=key[0],
                line=key[1],
                function=key[2],
                primitive_calls=value[0],
                calls=value[1],
                self_seconds=value[2],
                cumulative_seconds=value[3],
            )
            for key, value in profiler.stats.items()
        ]
        rows.sort(key=lambda row: row["cumulative_seconds"], reverse=True)
        (_diag_dir / f"processor-{_diag_uuid.uuid4().hex}.json").write_text(
            _diag_json.dumps(dict(start_ns=started, end_ns=_diag_time.time_ns(), functions=rows[:100]))
        )


@_diag_functools.wraps(_diag_preprocess_original)
async def _diag_preprocess(self, *args, **kwargs):
    started = _diag_time.time_ns()
    try:
        return await _diag_preprocess_original(self, *args, **kwargs)
    finally:
        (_diag_dir / f"preprocess-{_diag_uuid.uuid4().hex}.json").write_text(
            _diag_json.dumps(dict(start_ns=started, end_ns=_diag_time.time_ns()))
        )


_DiagProcessor.apply = _diag_apply
OmniOpenAIServingChat._preprocess_chat = _diag_preprocess

from vllm.multimodal.media.connector import MediaConnector as _DiagMediaConnector


def _diag_wrap_media(original, name):
    @_diag_functools.wraps(original)
    def wrapped(*args, **kwargs):
        profiler = _diag_cprofile.Profile()
        started = _diag_time.time_ns()
        try:
            return profiler.runcall(original, *args, **kwargs)
        finally:
            profiler.create_stats()
            rows = [
                dict(
                    file=key[0],
                    line=key[1],
                    function=key[2],
                    calls=value[1],
                    self_seconds=value[2],
                    cumulative_seconds=value[3],
                )
                for key, value in profiler.stats.items()
            ]
            rows.sort(key=lambda row: row["cumulative_seconds"], reverse=True)
            (_diag_dir / f"media_{name}-{_diag_uuid.uuid4().hex}.json").write_text(
                _diag_json.dumps(dict(start_ns=started, end_ns=_diag_time.time_ns(), functions=rows[:80]))
            )

    return wrapped


for _diag_method in ("__init__", "_load_data_url"):
    setattr(
        _DiagMediaConnector, _diag_method, _diag_wrap_media(getattr(_DiagMediaConnector, _diag_method), _diag_method)
    )

_diag_loads_original = _diag_json.loads


@_diag_functools.wraps(_diag_loads_original)
def _diag_loads(*args, **kwargs):
    started = _diag_time.time_ns()
    result = _diag_loads_original(*args, **kwargs)
    ended = _diag_time.time_ns()
    if isinstance(result, dict) and "messages" in result and "model" in result:
        (_diag_dir / f"json_loads-{_diag_uuid.uuid4().hex}.json").write_text(
            _diag_json.dumps(dict(start_ns=started, end_ns=ended, input_bytes=len(args[0])))
        )
    return result


_diag_json.loads = _diag_loads
