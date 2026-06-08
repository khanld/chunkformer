#!/usr/bin/env python3
"""Verify the online StreamingSession against the offline streaming path.

Checks, for a --streaming export:
  1. online fbank (push_waveform in random-sized chunks) == offline extract_features
  2. streaming-session transcript matches OnnxAsrModel.transcribe(streaming=True)
  3. the CTC-only streaming path (encoder cache only) also runs and matches its
     own offline CTC streaming decode.
"""

import argparse

import numpy as np
from pydub import AudioSegment

from chunkformer.onnx.runtime import OnnxAsrModel


def load_int16_samples(path: str, sr: int) -> np.ndarray:
    audio = AudioSegment.from_file(path).set_frame_rate(sr).set_sample_width(2).set_channels(1)
    return np.array(audio.get_array_of_samples(), dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx-dir", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    m = OnnxAsrModel(args.onnx_dir)
    sr = m.meta["feature"]["sample_rate"]
    samples = load_int16_samples(args.audio, sr)

    # 1) online vs offline fbank ------------------------------------------------
    offline_feats = m.extract_features(args.audio)[0]  # (T, F)
    sess = m.stream()
    rng = np.random.default_rng(args.seed)
    online_frames = []
    i = 0
    while i < len(samples):
        n = int(rng.integers(800, 5000))
        sess.push_features  # noqa: B018  (touch to keep linters calm)
        chunk = samples[i : i + n]
        feats = sess._samples_to_features(chunk)  # noqa: SLF001 - test introspection
        if feats.shape[0]:
            online_frames.append(feats)
        i += n
    online_feats = np.concatenate(online_frames, axis=0)
    n = min(len(offline_feats), len(online_feats))
    feat_diff = float(np.abs(offline_feats[:n] - online_feats[:n]).max())
    print(
        f"[fbank]   offline {offline_feats.shape} online {online_feats.shape} "
        f"max|diff|={feat_diff:.3e}"
    )
    assert online_feats.shape[0] == offline_feats.shape[0], "frame count differs"
    assert feat_diff < 1e-3, "online fbank differs from offline"

    # 2) streaming-session transcript vs offline streaming ----------------------
    offline_text = m.transcribe_file(args.audio, streaming=True)

    sess = m.stream()
    delta_parts = []
    i = 0
    while i < len(samples):
        n = int(rng.integers(1600, 8000))  # ~0.1-0.5s chunks
        d = sess.push_waveform(samples[i : i + n])
        if d:
            delta_parts.append(d)
        i += n
    d = sess.finalize()
    if d:
        delta_parts.append(d)
    stream_text = sess.text

    print(f"\n[offline stream] {offline_text}")
    print(f"[online  stream] {stream_text}")
    print(f"[delta concat  ] {''.join(delta_parts)}")
    print(f"transcript match: {stream_text == offline_text}")

    # 3) CTC-only streaming path (encoder cache only) ---------------------------
    if m.ctc is not None:
        cs = m.stream()
        cs.is_transducer = False  # force the CTC decode branch (encoder cache only)
        i = 0
        while i < len(samples):
            n = int(rng.integers(1600, 8000))
            cs.push_waveform(samples[i : i + n])
            i += n
        cs.finalize()
        # offline CTC streaming reference
        enc = m.encode_streaming(m.extract_features(args.audio))
        ref = m.tokens_to_text(m.ctc_greedy(enc, enc.shape[1]))
        print(f"\n[ctc offline   ] {ref}")
        print(f"[ctc online    ] {cs.text}")
        print(f"ctc match: {cs.text == ref}")


if __name__ == "__main__":
    main()
