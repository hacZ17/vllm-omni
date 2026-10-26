# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""One fault-free HunyuanImage request on the expert-parallel deploy.

Process-kill coverage on ROCm uses the reliability overlay. This test keeps
the TP=4 expert-parallel path visible without repeating it for every kill.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.dfx.reliability.helpers import FaultScenario, scenario_params
from tests.dfx.reliability.test_hunyuan_image import HunyuanImage
from tests.helpers.mark import hardware_marks


class HunyuanImageEpBaseline(HunyuanImage):
    res = {"rocm": "MI325"}
    scenarios = [
        {
            "test_name": "hunyuan_image_dit_ep_baseline",
            "server_params": {
                "model": "tencent/HunyuanImage-3.0-Instruct",
                "stage_config_name": "hunyuan_image3_dit.yaml",
            },
        }
    ]


class _EpBaseline(FaultScenario):
    def run(self, model: Any, server: Any, client: Any, signal_name: str | None = None) -> None:
        model.warmup(client, server)


MODEL = HunyuanImageEpBaseline()

pytestmark = [
    pytest.mark.full_model,
    pytest.mark.slow,
    pytest.mark.diffusion,
    *hardware_marks(res=MODEL.res, num_cards=MODEL.num_cards),
]


@scenario_params(MODEL, _EpBaseline)
def test_hunyuan_image_rocm_ep_baseline(omni_server_function, online_client_function) -> None:
    """Send one 512x512 image request with expert parallel left on."""
    _EpBaseline().run(MODEL, omni_server_function, online_client_function)
