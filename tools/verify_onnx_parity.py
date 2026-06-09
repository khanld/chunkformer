#!/usr/bin/env python3
"""Compare exported ONNX graphs against the PyTorch ChunkFormer model.

For each module it reports the maximum absolute numerical difference between the
PyTorch op and its ONNX Runtime counterpart (fed identical inputs), and it prints
the final decoded transcript from both backends.

Examples
--------
    # Single model
    python tools/verify_onnx_parity.py \\
        --checkpoint khanhld/chunkformer-ctc-large-vie \\
        --onnx-dir onnx_out/ctc --audio samples/audios/audio_1.wav

    # All three reference checkpoints (expects onnx_out/{ctc,rnnt,rnnt_stream})
    python tools/verify_onnx_parity.py --all --audio samples/audios/audio_1.wav
"""

import argparse
from typing import List

import numpy as np
import torch

from chunkformer.chunkformer_model import ChunkFormerModel
from chunkformer.onnx.runtime import OnnxAsrModel, remove_duplicates_and_blank
from chunkformer.transducer.search.greedy_search import basic_greedy_search


def _maxdiff(a: np.ndarray, b: np.ndarray) -> float:
    n = min(a.shape[1], b.shape[1]) if a.ndim >= 2 else min(a.shape[0], b.shape[0])
    if a.ndim >= 2:
        a, b = a[:, :n], b[:, :n]
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


@torch.no_grad()
def verify(checkpoint: str, onnx_dir: str, audio: str, device: str = "cpu") -> None:
    print("=" * 78)
    print(f"Checkpoint : {checkpoint}")
    print(f"ONNX dir   : {onnx_dir}")
    print(f"Audio      : {audio}")
    print("-" * 78)

    pt = ChunkFormerModel.from_pretrained(checkpoint)
    pt.eval()
    inner = pt.model
    onnx = OnnxAsrModel(onnx_dir, device=device, char_dict=pt.char_dict)
    is_transducer = onnx.meta.get("is_transducer", False)
    streaming = onnx.meta.get("streaming", False)

    feats, flen = pt._load_audio_and_extract_features(audio)
    xs = feats.unsqueeze(0)
    xs_lens = torch.tensor([flen], dtype=torch.int64)

    # ---- 1. non-streaming encoder (full-context, or limited-context if baked)
    fc = onnx.meta.get("full_context", {})
    fc_chunk = fc.get("chunk_size", 0)
    fc_left = fc.get("left_context_size", 0)
    fc_right = fc.get("right_context_size", 0)
    enc_pt, mask_pt = inner.encoder.forward_encoder(xs, xs_lens, fc_chunk, fc_left, fc_right)
    enc_len_pt = int(mask_pt.squeeze(1).sum(1)[0])
    enc_onnx, enc_lens_onnx = onnx.encode_full(xs.numpy(), xs_lens.numpy())
    ctx = "full" if fc_chunk == 0 else f"chunk={fc_chunk} l={fc_left} r={fc_right}"
    print(
        f"[encoder_full]  ({ctx})  out shape {tuple(enc_pt.shape)}  "
        f"len pt={enc_len_pt} onnx={int(enc_lens_onnx[0])}  "
        f"max|diff|={_maxdiff(enc_pt.numpy(), enc_onnx):.3e}"
    )

    # ---- 2. CTC head (feed identical PyTorch encoder output)
    if onnx.ctc is not None:
        ctc_pt = inner.ctc.log_softmax(enc_pt).numpy()
        ctc_onnx = onnx.ctc_logprobs(enc_pt.numpy())
        print(
            f"[ctc]           out shape {ctc_pt.shape[1:]}  "
            f"max|diff|={_maxdiff(ctc_pt, ctc_onnx):.3e}"
        )

    # ---- 3 & 4. RNN-T predictor + joint
    if is_transducer:
        assert onnx.predictor is not None and onnx.joint is not None
        n_layers = onnx.meta["predictor"]["n_layers"]
        hidden = onnx.meta["predictor"]["hidden_size"]
        token = torch.zeros(1, 1, dtype=torch.int64)
        m = torch.zeros(n_layers, 1, hidden)
        c = torch.zeros(n_layers, 1, hidden)
        pred_pt, (nm_pt, nc_pt) = inner.predictor.forward_step(token, [m, c])
        pred_onnx, nm_onnx, nc_onnx = onnx.predictor.run(
            ["pred_out", "new_m", "new_c"],
            {"token": token.numpy(), "state_m": m.numpy(), "state_c": c.numpy()},
        )
        print(
            f"[predictor]     out shape {tuple(pred_pt.shape)}  "
            f"max|diff|={_maxdiff(pred_pt.numpy(), pred_onnx):.3e}"
        )

        enc_t = enc_pt[:, :1, :]
        joint_pt = inner.joint(enc_t, pred_pt).numpy()
        (joint_onnx,) = onnx.joint.run(
            ["logits"], {"enc_t": enc_t.numpy(), "pred": pred_pt.numpy()}
        )
        print(
            f"[joint]         out shape {joint_pt.shape[1:]}  "
            f"max|diff|={_maxdiff(joint_pt, joint_onnx):.3e}"
        )

    # ---- 5. streaming chunk encoder
    if streaming and onnx.encoder_chunk is not None:
        s = onnx.meta["stream"]
        enc_stream_pt, mask_s = inner.encoder.forward_chunk_by_chunk(
            xs, xs_lens, s["chunk_size"], s["left_context_size"], s["right_context_size"]
        )
        enc_stream_onnx = onnx.encode_streaming(xs.numpy())
        print(
            f"[encoder_chunk] out shape {tuple(enc_stream_pt.shape)} (pt) / "
            f"{enc_stream_onnx.shape} (onnx)  "
            f"max|diff|={_maxdiff(enc_stream_pt.numpy(), enc_stream_onnx):.3e}"
        )

    # ---- text output
    pt_tokens = _pt_tokens(inner, enc_pt, enc_len_pt, is_transducer, onnx.blank_id)
    pt_text = onnx.tokens_to_text(pt_tokens)
    onnx_text = onnx.transcribe(xs.numpy(), xs_lens.numpy(), streaming=False)
    print("-" * 78)
    print(f"PyTorch text : {pt_text}")
    print(f"ONNX text    : {onnx_text}")
    print(f"Text match   : {pt_text == onnx_text}")
    if streaming and onnx.encoder_chunk is not None:
        onnx_stream_text = onnx.transcribe(xs.numpy(), xs_lens.numpy(), streaming=True)
        print(f"ONNX (stream): {onnx_stream_text}")
    print("=" * 78)
    print()


def _pt_tokens(
    inner, enc_pt: torch.Tensor, enc_len: int, is_transducer: bool, blank_id: int
) -> List[int]:
    if is_transducer:
        return basic_greedy_search(inner, enc_pt, torch.tensor(enc_len))[0]
    log_probs = inner.ctc.log_softmax(enc_pt)[0][:enc_len]
    hyp = log_probs.argmax(dim=-1).tolist()
    return remove_duplicates_and_blank(hyp, blank_id)


REFERENCE = [
    ("khanhld/chunkformer-ctc-large-vie", "onnx_out/ctc"),
    ("khanhld/chunkformer-rnnt-large-vie", "onnx_out/rnnt"),
    ("khanhld/chunkformer-rnnt-small-vie-stream-dct", "onnx_out/rnnt_stream"),
]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", help="Local dir or HF repo id of the model")
    parser.add_argument("--onnx-dir", help="Directory with the exported ONNX graphs")
    parser.add_argument("--audio", default="samples/audios/audio_1.wav", help="Sample audio path")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--all", action="store_true", help="Run the 3 reference checkpoints")
    args = parser.parse_args()

    if args.all:
        for ckpt, d in REFERENCE:
            verify(ckpt, d, args.audio, args.device)
    else:
        if not args.checkpoint or not args.onnx_dir:
            parser.error("--checkpoint and --onnx-dir are required unless --all is given")
        verify(args.checkpoint, args.onnx_dir, args.audio, args.device)


if __name__ == "__main__":
    main()
