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
        # Fall back to the vocab.txt written next to the graphs so text output
        # needs no external files / no PyTorch model.
        if char_dict is None:
            char_dict = self._load_vocab(os.path.join(model_dir, "vocab.txt"))
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

    @staticmethod
    def _load_vocab(path: str) -> Optional[Dict[int, str]]:
        """Read a ``token id`` per-line vocab file into ``{id: token}``."""
        if not os.path.exists(path):
            return None
        char_dict: Dict[int, str] = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                arr = line.strip().split()
                if len(arr) == 2:
                    char_dict[int(arr[1])] = arr[0]
        return char_dict or None

    # -------------------------------------------------------------- features
    def extract_features(self, audio_path: str) -> np.ndarray:
        """Waveform -> raw fbank (B=1, T, F) float32, matching the PyTorch model.

        CMVN is baked into the encoder graph, so no normalisation is applied
        here. Requires ``torch``/``torchaudio`` and ``pydub`` (lazy-imported).
        """
        try:
            import torch
            import torchaudio.compliance.kaldi as kaldi
            from pydub import AudioSegment
        except ImportError as e:  # pragma: no cover - optional deps
            raise ImportError(
                "extract_features needs torch, torchaudio and pydub. Either install "
                "them, or pass pre-computed fbank features to transcribe()/encode_*()."
            ) from e

        feat = self.meta.get("feature", {})
        sample_rate = int(feat.get("sample_rate", 16000))
        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_frame_rate(sample_rate).set_sample_width(2).set_channels(1)
        waveform = torch.as_tensor(audio.get_array_of_samples(), dtype=torch.float32).unsqueeze(0)
        mat = kaldi.fbank(
            waveform,
            num_mel_bins=int(feat.get("num_mel_bins", self.meta["feat_dim"])),
            frame_length=int(feat.get("frame_length", 25)),
            frame_shift=int(feat.get("frame_shift", 10)),
            dither=0.0,
            energy_floor=0.0,
            sample_frequency=sample_rate,
        )
        feats: np.ndarray = mat.numpy()[None].astype(np.float32)
        return feats

    def transcribe_file(self, audio_path: str, streaming: bool = False) -> str:
        """End-to-end: audio file path -> text (self-contained)."""
        feats = self.extract_features(audio_path)
        feat_lens = np.array([feats.shape[1]], dtype=np.int64)
        return self.transcribe(feats, feat_lens, streaming=streaming)

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

    # ----------------------------------------------------------------- streaming
    def stream(self, n_steps: int = 64) -> "StreamingSession":
        """Open a stateful online streaming session.

        Feed audio incrementally with ``push_waveform`` / ``push_features`` and
        get the text decoded so far for each push; call ``finalize`` to flush.
        Manages the encoder caches (CTC and RNN-T) and, for RNN-T, the predictor
        LSTM state across chunks. Requires a ``--streaming`` export.
        """
        return StreamingSession(self, n_steps=n_steps)


class StreamingSession:
    """Online, cache-aware streaming decode for one utterance.

    Audio (or pre-computed fbank) is pushed incrementally. Internally it:

    * extracts fbank frames online (sample buffer with kaldi snip-edges framing),
    * runs ``encoder_chunk.onnx`` once per ``size``-frame window, threading the
      attention and convolution caches plus the warm-up ``offset``,
    * decodes the newly finalized encoder frames with CTC or RNN-T greedy search,
      carrying the CTC repeat-merge state or the RNN-T predictor LSTM state and
      last emitted token across chunks.

    Each ``push_*`` / ``finalize`` returns the *delta* text produced by that call.
    The full hypothesis is available via ``.text`` and ``.tokens``.
    """

    def __init__(self, model: OnnxAsrModel, n_steps: int = 64):
        if model.encoder_chunk is None or "stream" not in model.meta:
            raise ValueError(
                "StreamingSession requires a streaming export (run export_onnx.py "
                "with --streaming)."
            )
        self.m = model
        self.n_steps = n_steps
        stream = model.meta["stream"]
        self.chunk_size = int(stream["chunk_size"])
        self.left = int(stream["left_context_size"])
        self.size = int(stream["size"])
        self.stride = int(stream["stride"])

        nb = int(model.meta["num_blocks"])
        heads = int(model.meta["attention_heads"])
        d_out = int(model.meta["encoder_output_size"])
        lorder = int(model.meta["cnn_lorder"])
        self.att_cache = np.zeros((nb, 1, heads, self.left, d_out // heads * 2), dtype=np.float32)
        self.cnn_cache = np.zeros((nb, 1, d_out, lorder), dtype=np.float32)
        self.offset = 0
        self.is_transducer = bool(model.meta.get("is_transducer"))

        # feature-extraction (online fbank) state
        feat = model.meta.get("feature", {})
        self.sr = int(feat.get("sample_rate", 16000))
        self.num_mel_bins = int(feat.get("num_mel_bins", model.meta["feat_dim"]))
        self.frame_length_ms = int(feat.get("frame_length", 25))
        self.frame_shift_ms = int(feat.get("frame_shift", 10))
        self._win = int(round(self.sr * self.frame_length_ms / 1000))
        self._shift = int(round(self.sr * self.frame_shift_ms / 1000))
        self._sample_buf = np.zeros(0, dtype=np.float32)
        self._feat_buf = np.zeros((0, model.meta["feat_dim"]), dtype=np.float32)

        # decode state
        self.tokens: List[int] = []
        self._ctc_prev = -1  # last argmax id, for cross-chunk CTC repeat-merge
        if self.is_transducer:
            self._init_rnnt_state()

    # -------------------------------------------------------------- public API
    @property
    def text(self) -> str:
        return self.m.tokens_to_text(self.tokens)

    def push_features(self, frames: np.ndarray) -> str:
        """Push raw fbank frames (n, F). Returns the delta text."""
        prev = self.text
        if frames is not None and frames.shape[0] > 0:
            self._feat_buf = np.concatenate([self._feat_buf, frames.astype(np.float32)], axis=0)
        while self._feat_buf.shape[0] >= self.size:
            window = self._feat_buf[: self.size][None]
            enc = self._run_encoder(window)[:, : self.chunk_size, :]
            self._decode(enc)
            self._feat_buf = self._feat_buf[self.stride :]
        return self.text[len(prev) :]

    def push_waveform(self, samples: np.ndarray) -> str:
        """Push int16-range PCM samples (1-D). Returns the delta text.

        Mirrors the offline extractor: raw int16 sample values, no normalisation,
        kaldi snip-edges framing. Requires torch/torchaudio.
        """
        return self.push_features(self._samples_to_features(samples))

    def finalize(self) -> str:
        """Flush the remaining buffered frames as the final (padded) window."""
        prev = self.text
        if self._feat_buf.shape[0] > 0:
            window = self._feat_buf
            if window.shape[0] < self.size:
                window = np.pad(window, ((0, self.size - window.shape[0]), (0, 0)))
            enc = self._run_encoder(window[None])  # keep all frames on the last window
            self._decode(enc)
            self._feat_buf = self._feat_buf[:0]
        return self.text[len(prev) :]

    # ---------------------------------------------------------------- internals
    def _run_encoder(self, window: np.ndarray) -> np.ndarray:
        assert self.m.encoder_chunk is not None
        enc_out, self.att_cache, self.cnn_cache = self.m.encoder_chunk.run(
            ["enc_out", "r_att_cache", "r_cnn_cache"],
            {
                "chunk": window.astype(np.float32),
                "att_cache": self.att_cache,
                "cnn_cache": self.cnn_cache,
                "offset": np.array(self.offset, dtype=np.int64),
            },
        )
        self.offset += self.chunk_size
        return np.asarray(enc_out)

    def _decode(self, enc: np.ndarray) -> None:
        if self.is_transducer:
            self._decode_rnnt(enc)
        else:
            self._decode_ctc(enc)

    def _decode_ctc(self, enc: np.ndarray) -> None:
        log_probs = self.m.ctc_logprobs(enc)[0]  # (T, V)
        for tok in log_probs.argmax(axis=-1).tolist():
            if tok != self._ctc_prev:
                if tok != self.m.blank_id:
                    self.tokens.append(int(tok))
                self._ctc_prev = tok

    # --- RNN-T persistent state ------------------------------------------------
    def _init_rnnt_state(self) -> None:
        assert self.m.predictor is not None and self.m.joint is not None
        n_layers = int(self.m.meta["predictor"]["n_layers"])
        hidden = int(self.m.meta["predictor"]["hidden_size"])
        self._state_m = np.zeros((n_layers, 1, hidden), dtype=np.float32)
        self._state_c = np.zeros((n_layers, 1, hidden), dtype=np.float32)
        self._token = np.array([[self.m.blank_id]], dtype=np.int64)
        self._prev_nblk = True
        self._pred_out, self._cand_m, self._cand_c = self.m.predictor.run(
            ["pred_out", "new_m", "new_c"],
            {"token": self._token, "state_m": self._state_m, "state_c": self._state_c},
        )

    def _decode_rnnt(self, enc: np.ndarray) -> None:
        assert self.m.predictor is not None and self.m.joint is not None
        for t in range(enc.shape[1]):
            enc_t = enc[:, t : t + 1, :].astype(np.float32)
            per_frame_noblk = 0
            while True:
                if self._prev_nblk:
                    self._pred_out, self._cand_m, self._cand_c = self.m.predictor.run(
                        ["pred_out", "new_m", "new_c"],
                        {
                            "token": self._token,
                            "state_m": self._state_m,
                            "state_c": self._state_c,
                        },
                    )
                (logits,) = self.m.joint.run(["logits"], {"enc_t": enc_t, "pred": self._pred_out})
                best = int(logits.reshape(-1).argmax())
                if best != self.m.blank_id:
                    self.tokens.append(best)
                    self._prev_nblk = True
                    per_frame_noblk += 1
                    self._token = np.array([[best]], dtype=np.int64)
                    self._state_m, self._state_c = self._cand_m, self._cand_c
                if best == self.m.blank_id or per_frame_noblk >= self.n_steps:
                    self._prev_nblk = best != self.m.blank_id
                    break

    # --- online fbank ----------------------------------------------------------
    def _samples_to_features(self, samples: np.ndarray) -> np.ndarray:
        try:
            import torch
            import torchaudio.compliance.kaldi as kaldi
        except ImportError as e:  # pragma: no cover - optional deps
            raise ImportError(
                "push_waveform needs torch and torchaudio. Use push_features with "
                "pre-computed fbank for a torch-free deployment."
            ) from e

        self._sample_buf = np.concatenate(
            [self._sample_buf, np.asarray(samples, dtype=np.float32).reshape(-1)]
        )
        if self._sample_buf.shape[0] < self._win:
            return np.zeros((0, self._feat_buf.shape[1]), dtype=np.float32)
        n_frames = 1 + (self._sample_buf.shape[0] - self._win) // self._shift
        end = (n_frames - 1) * self._shift + self._win
        wav = torch.from_numpy(self._sample_buf[:end].copy()).float().unsqueeze(0)
        mat = kaldi.fbank(
            wav,
            num_mel_bins=self.num_mel_bins,
            frame_length=self.frame_length_ms,
            frame_shift=self.frame_shift_ms,
            dither=0.0,
            energy_floor=0.0,
            sample_frequency=self.sr,
        )
        feats: np.ndarray = mat.numpy().astype(np.float32)
        self._sample_buf = self._sample_buf[n_frames * self._shift :]
        return feats
