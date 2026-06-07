# ChunkFormer ONNX Export & Inference

Export ChunkFormer ASR models (CTC and RNN-T) to standalone ONNX graphs and run
them with ONNX Runtime. Both **full-context** (non-streaming) and **cache-aware
streaming** encoder paths are supported. Search/decoding loops stay in host
Python; only the neural-network sub-graphs run in ONNX.

## Install

```bash
pip install -e ".[onnx]"        # adds onnx, onnxruntime, onnxsim
# or, for GPU inference:  pip install onnxruntime-gpu
```

## Exported graphs

`tools/export_onnx.py` writes one ONNX file per submodule plus an
`onnx_config.json` with the shapes/metadata the runtime needs:

| File | Module | Inputs | Outputs |
|---|---|---|---|
| `encoder_full.onnx` | full-context encoder | `feats[B,T,80]`, `feat_lens[B]` | `enc_out[B,T',D]`, `enc_lens[B]` |
| `encoder_chunk.onnx` | streaming encoder (`--streaming`) | `chunk`, `att_cache`, `cnn_cache`, `offset` | `enc_out`, `r_att_cache`, `r_cnn_cache` |
| `ctc.onnx` | CTC head | `enc_out[B,T,D]` | `log_probs[B,T,V]` |
| `predictor.onnx` | RNN-T predictor step | `token[B,1]`, `state_m`, `state_c` | `pred_out`, `new_m`, `new_c` |
| `joint.onnx` | RNN-T joint | `enc_t[B,1,E]`, `pred[B,1,P]` | `logits[B,1,V]` |

CMVN is baked into the encoder graph, so ONNX consumes raw 80-dim fbank features.

## Export

```bash
# CTC model (non-streaming)
python tools/export_onnx.py \
    --checkpoint khanhld/chunkformer-ctc-large-vie --out-dir onnx_out/ctc

# RNN-T model (non-streaming)
python tools/export_onnx.py \
    --checkpoint khanhld/chunkformer-rnnt-large-vie --out-dir onnx_out/rnnt

# Streaming RNN-T model (cache-aware). chunk/left/right must be a configuration
# the model was trained with (see the model's dynamic_chunk_sizes).
python tools/export_onnx.py \
    --checkpoint khanhld/chunkformer-rnnt-small-vie-stream-dct --streaming \
    --chunk-size 8 --left-context 60 --right-context 0 --out-dir onnx_out/rnnt_stream
```

`--checkpoint` accepts a local directory or a Hugging Face Hub repo id. Add
`--simplify` to run `onnxsim` on each graph.

### Non-streaming limited-context (masked chunking)

The non-streaming encoder also supports ChunkFormer's masked limited-context
decoding (efficient long-form mode). Pass `--full-chunk-size` (>0) with
`--full-left-context` / `--full-right-context` to bake a chunk configuration into
`encoder_full.onnx` instead of full self-attention:

```bash
python tools/export_onnx.py \
    --checkpoint khanhld/chunkformer-ctc-large-vie \
    --full-chunk-size 64 --full-left-context 128 --full-right-context 4 \
    --out-dir onnx_out/ctc_limited
```

The time dimension stays dynamic, so one graph handles variable-length audio.
`--full-chunk-size 0` (default) keeps full-context self-attention.

## Inference

```python
import numpy as np
from chunkformer.chunkformer_model import ChunkFormerModel
from chunkformer.onnx.runtime import OnnxAsrModel

# char_dict (id -> token) for text output; reuse the PyTorch model's table
pt = ChunkFormerModel.from_pretrained("khanhld/chunkformer-ctc-large-vie")
feats, flen = pt._load_audio_and_extract_features("samples/audios/audio_1.wav")

onnx = OnnxAsrModel("onnx_out/ctc", device="cpu", char_dict=pt.char_dict)
text = onnx.transcribe(
    feats.unsqueeze(0).numpy(), np.array([flen], dtype=np.int64), streaming=False
)
print(text)
```

For a streaming model, pass `streaming=True` (the runtime runs the cache-aware
chunk loop using the parameters stored in `onnx_config.json`).

## Verify parity

`tools/verify_onnx_parity.py` reports the max absolute difference between each
PyTorch op and its ONNX counterpart, plus the decoded transcript from both:

```bash
python tools/verify_onnx_parity.py --all --audio samples/audios/audio_1.wav
```

Reference results (CPU, fp32, opset 17) on `audio_1.wav`:

| Model | encoder_full | ctc | predictor | joint | encoder_chunk | text match |
|---|---|---|---|---|---|---|
| `chunkformer-ctc-large-vie` | 4.5e-6 | 1.5e-5 | - | - | - | yes |
| `chunkformer-rnnt-large-vie` | 1.7e-6 | 9.5e-6 | 7.2e-7 | 4.8e-6 | - | yes |
| `chunkformer-rnnt-small-vie-stream-dct` | 3.3e-6 | 1.9e-5 | 1.4e-6 | 5.7e-6 | 6.9e-6 | yes |

All differences are at float32 numerical-noise level and transcripts match
exactly between PyTorch and ONNX.

## Notes / limitations

- The attention (AED) decoder is **not** exported; CTC and RNN-T greedy decoding
  are supported. Beam search / attention rescoring remain PyTorch-only.
- The exporter uses the legacy TorchScript exporter (`dynamo=False`). Two ops are
  ONNX-incompatible in their native form and are transparently rewritten only
  during export (guarded by `torch.onnx.is_in_onnx_export()`), leaving eager
  training/inference unchanged:
  - relative-attention `rel_shift` (`as_strided` -> gather);
  - chunking windows in limited-context attention and dynamic-conv
    (`Tensor.unfold` -> gather / unsqueeze, via `chunkformer.utils.mask.onnx_unfold`).
- Streaming `--chunk-size/--left-context/--right-context` must match a
  configuration the model was trained with.
