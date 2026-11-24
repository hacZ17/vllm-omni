# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Diagnostic only: appended inside the pod to serving_chat.py, never merged.
import cProfile as _diag_cprofile  # noqa: N813
import functools as _diag_functools
import json as _diag_json
import os as _diag_os
import pstats as _diag_pstats
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
        stats = _diag_pstats.Stats(profiler)
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
            for key, value in stats.stats.items()
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
