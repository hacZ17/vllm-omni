# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Diagnostic only. Preserve the client's serializer and measurement boundary.
import functools as _json_diag_functools
import os as _json_diag_os
import time as _json_diag_time
import uuid as _json_diag_uuid
from pathlib import Path as _JsonDiagPath

_json_diag_session_init = aiohttp.ClientSession.__init__


@_json_diag_functools.wraps(_json_diag_session_init)
def _json_diag_init(self, *args, **kwargs):
    _json_diag_session_init(self, *args, **kwargs)
    original = self._json_serialize

    def measured(payload):
        started = _json_diag_time.time_ns()
        encoded = original(payload)
        ended = _json_diag_time.time_ns()
        if isinstance(payload, dict) and "messages" in payload:
            path = _JsonDiagPath(_json_diag_os.environ["QWEN_FRONTEND_PROFILE_DIR"])
            path.mkdir(parents=True, exist_ok=True)
            (path / f"json_dumps-{_json_diag_uuid.uuid4().hex}.json").write_text(
                json.dumps(dict(start_ns=started, end_ns=ended, output_bytes=len(encoded)))
            )
        return encoded

    self._json_serialize = measured


aiohttp.ClientSession.__init__ = _json_diag_init
