# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Diagnostic prototype only: appended to the Code2Wav module in the pod.
import time as _warm_time

_original_capture_auxiliary = Qwen3TTSCode2Wav.capture_auxiliary_graphs


def _warm_eager_decoder(self):
    _original_capture_auxiliary(self)
    if not self._async_chunk:
        return
    decoder = self.decoder
    device = next(decoder.parameters()).device
    initial = int(getattr(decoder, "_initial_codec_chunk_frames", 1))
    chunk = int(getattr(decoder, "_incremental_chunk_frames", 25))
    prefix = int(decoder.config.sliding_window)
    ramp = list(getattr(decoder, "_incremental_chunk_ramp", ()) or ())
    assert not ramp, "Prototype must not silently omit a configured ramp"
    assert 0 < self._decode_batch_max_size <= 4, "Prototype bounded to the observed deployment"
    started = _warm_time.perf_counter()
    with torch.inference_mode():
        for batch in range(1, self._decode_batch_max_size + 1):
            for mode in self.decoder_cudagraph_modes:
                ref_frames = prefix if mode == "icl" else 0
                first_flags = [False]
                if mode == "xvec" and getattr(decoder, "capture_first_audio_state_only", False):
                    first_flags.append(True)
                for skip_first in first_flags:
                    caches = [{"prefix_frames": ref_frames, "skip_first_audio": skip_first} for _ in range(batch)]
                    # Exercise the prefix, every growth phase and two rolling
                    # suffixes through the exact eager request path.
                    lengths = [ref_frames + initial] + [chunk] * ((prefix + chunk - 1) // chunk + 2)
                    for frames in lengths:
                        codes = torch.zeros(
                            batch, decoder.config.num_quantizers, frames, dtype=torch.long, device=device
                        )
                        decoder.batched_chunked_decode(codes, [frames] * batch, caches=caches, max_batch_size=batch)
    torch.accelerator.synchronize(device)
    logger.info("DIAGNOSTIC eager decoder warmup completed in %.3fs", _warm_time.perf_counter() - started)


Qwen3TTSCode2Wav.capture_auxiliary_graphs = _warm_eager_decoder
