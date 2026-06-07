"""ONNX Runtime inference for exported ChunkFormer models.

Loads the per-submodule ONNX graphs produced by ``tools/export_onnx.py`` and
runs host-side CTC / RNN-T greedy decoding. The neural network sub-graphs run in
ONNX Runtime; all search/loop logic stays in Python (mirroring
``chunkformer.modules.search`` and ``chunkformer.transducer.search``).
"""

import json
import os
from typing import Dict, List, Optional

import numpy as np

try:
    import onnxruntime as ort
except ImportError as e:  # pragma: no cover - optional dependency
    raise ImportError(
        "onnxruntime is required for chunkformer.onnx.runtime. "
        "Install it with `pip install onnxruntime` (or the [onnx] extra)."
    ) from e


def _providers(device: str) -> List[str]:
    if device == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def remove_duplicates_and_blank(hyp: List[int], blank_id: int = 0) -> List[int]:
    """CTC collapse: merge repeats then drop blanks."""
    out: List[int] = []
    prev = -1
    for tok in hyp:
        if tok != prev:
            if tok != blank_id:
                out.append(tok)
            prev = tok
    return out


class OnnxAsrModel:
    """Run an exported ChunkFormer model with ONNX Runtime.

    Args:
        model_dir: directory containing the exported ``*.onnx`` graphs and
            ``onnx_config.json``.
        device: ``"cpu"`` or ``"cuda"``.
        char_dict: optional ``{id: token}`` map for producing text output.
    """

    def __init__(
        self,
        model_dir: str,
        device: str = "cpu",
        char_dict: Optional[Dict[int, str]] = None,
    ):
        self.model_dir = model_dir
        self.device = device
        with open(os.path.join(model_dir, "onnx_config.json"), "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        self.blank_id = int(self.meta.get("blank_id", 0))
        self.char_dict = char_dict
        providers = _providers(device)

        def _load(name: str) -> Optional[ort.InferenceSession]:
            path = os.path.join(model_dir, name)
            if not os.path.exists(path):
                return None
            return ort.InferenceSession(path, providers=providers)

        self.encoder_full = _load("encoder_full.onnx")
        self.encoder_chunk = _load("encoder_chunk.onnx")
        self.ctc = _load("ctc.onnx")
        self.predictor = _load("predictor.onnx")
        self.joint = _load("joint.onnx")

    # ------------------------------------------------------------------ encoder
    def encode_full(self, feats: np.ndarray, feat_lens: np.ndarray):
        """Full-context encoder. feats: (B, T, F) float32."""
        assert self.encoder_full is not None, "encoder_full.onnx not found"
        enc_out, enc_lens = self.encoder_full.run(
            ["enc_out", "enc_lens"],
            {"feats": feats.astype(np.float32), "feat_lens": feat_lens.astype(np.int64)},
        )
        return enc_out, enc_lens

    def encode_streaming(self, feats: np.ndarray):
        """Cache-aware streaming encoder over a single utterance.

        feats: (1, T, F) float32. Mirrors ``encoder.forward_chunk_by_chunk``.
        Returns enc_out (1, T', D).
        """
        assert self.encoder_chunk is not None, "encoder_chunk.onnx not found"
        stream = self.meta["stream"]
        chunk_size = stream["chunk_size"]
        left = stream["left_context_size"]
        size = stream["size"]
        stride = stream["stride"]
        num_blocks = self.meta["num_blocks"]
        heads = self.meta["attention_heads"]
        d_out = self.meta["encoder_output_size"]
        conv_lorder = self.meta["cnn_lorder"]

        x = feats.astype(np.float32)
        # pad so the last window is complete (mirrors forward_chunk_by_chunk)
        t = x.shape[1]
        if t >= size:
            pad = (stride - ((t - size) % stride)) % stride
        else:
            pad = size - t
        if pad > 0:
            x = np.pad(x, ((0, 0), (0, pad), (0, 0)))

        att_cache = np.zeros((num_blocks, 1, heads, left, d_out // heads * 2), dtype=np.float32)
        cnn_cache = np.zeros((num_blocks, 1, d_out, conv_lorder), dtype=np.float32)

        outs = []
        offset = 0
        total = x.shape[1]
        for i in range(0, total - size + stride, stride):
            chunk = x[:, i : i + size, :]
            enc_out, att_cache, cnn_cache = self.encoder_chunk.run(
                ["enc_out", "r_att_cache", "r_cnn_cache"],
                {
                    "chunk": chunk,
                    "att_cache": att_cache,
                    "cnn_cache": cnn_cache,
                    "offset": np.array(offset, dtype=np.int64),
                },
            )
            if i + size < total:
                enc_out = enc_out[:, :chunk_size, :]
            outs.append(enc_out)
            offset += chunk_size
        return np.concatenate(outs, axis=1)

    # --------------------------------------------------------------------- CTC
    def ctc_logprobs(self, enc_out: np.ndarray) -> np.ndarray:
        assert self.ctc is not None, "ctc.onnx not found"
        (log_probs,) = self.ctc.run(["log_probs"], {"enc_out": enc_out.astype(np.float32)})
        return np.asarray(log_probs)

    def ctc_greedy(self, enc_out: np.ndarray, enc_len: int) -> List[int]:
        log_probs = self.ctc_logprobs(enc_out)[0][:enc_len]
        hyp = log_probs.argmax(axis=-1).tolist()
        return remove_duplicates_and_blank(hyp, self.blank_id)

    # ------------------------------------------------------------------- RNN-T
    def rnnt_greedy(self, enc_out: np.ndarray, enc_len: int, n_steps: int = 64) -> List[int]:
        """Single-utterance RNN-T greedy search (mirrors basic_greedy_search)."""
        assert self.predictor is not None and self.joint is not None
        n_layers = self.meta["predictor"]["n_layers"]
        hidden = self.meta["predictor"]["hidden_size"]

        state_m = np.zeros((n_layers, 1, hidden), dtype=np.float32)
        state_c = np.zeros((n_layers, 1, hidden), dtype=np.float32)
        token = np.array([[self.blank_id]], dtype=np.int64)

        pred_out, new_m, new_c = self.predictor.run(
            ["pred_out", "new_m", "new_c"],
            {"token": token, "state_m": state_m, "state_c": state_c},
        )

        hyps: List[int] = []
        t = 0
        prev_nblk = True
        per_frame_noblk = 0
        while t < enc_len:
            enc_t = enc_out[:, t : t + 1, :].astype(np.float32)
            if prev_nblk:
                pred_out, new_m, new_c = self.predictor.run(
                    ["pred_out", "new_m", "new_c"],
                    {"token": token, "state_m": state_m, "state_c": state_c},
                )
            (logits,) = self.joint.run(["logits"], {"enc_t": enc_t, "pred": pred_out})
            best = int(logits.reshape(-1).argmax())
            if best != self.blank_id:
                hyps.append(best)
                prev_nblk = True
                per_frame_noblk += 1
                token = np.array([[best]], dtype=np.int64)
                state_m, state_c = new_m, new_c
            if best == self.blank_id or per_frame_noblk >= n_steps:
                prev_nblk = best != self.blank_id
                t += 1
                per_frame_noblk = 0
        return hyps

    # ------------------------------------------------------------------- decode
    def tokens_to_text(self, tokens: List[int]) -> str:
        if self.char_dict is None:
            return " ".join(str(t) for t in tokens)
        pieces = [self.char_dict.get(t, "") for t in tokens]
        return "".join(pieces).replace("\u2581", " ").strip()

    def transcribe(self, feats: np.ndarray, feat_lens: np.ndarray, streaming: bool = False) -> str:
        if streaming:
            enc_out = self.encode_streaming(feats)
            enc_len = enc_out.shape[1]
        else:
            enc_out, enc_lens = self.encode_full(feats, feat_lens)
            enc_len = int(enc_lens[0])
        if self.meta.get("is_transducer"):
            tokens = self.rnnt_greedy(enc_out, enc_len)
        else:
            tokens = self.ctc_greedy(enc_out, enc_len)
        return self.tokens_to_text(tokens)
