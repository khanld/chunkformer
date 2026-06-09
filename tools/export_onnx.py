#!/usr/bin/env python3
"""Export a ChunkFormer ASR model (CTC or RNN-T) to ONNX.

Each network submodule is exported as a separate ONNX graph:

- ``encoder_full.onnx``   full-context (non-streaming) encoder.
- ``encoder_chunk.onnx``  cache-aware streaming encoder (``--streaming``).
- ``ctc.onnx``            CTC head (if the model has one).
- ``predictor.onnx``      RNN-T predictor step (transducer models).
- ``joint.onnx``          RNN-T joint network (transducer models).

Search / decoding loops stay in host Python (see ``chunkformer.onnx.runtime``).

Examples
--------
    # CTC model, non-streaming
    python tools/export_onnx.py \\
        --checkpoint khanhld/chunkformer-ctc-large-vie --out-dir onnx/ctc

    # RNN-T model, non-streaming
    python tools/export_onnx.py \\
        --checkpoint khanhld/chunkformer-rnnt-large-vie --out-dir onnx/rnnt

    # Streaming RNN-T model (cache-aware)
    python tools/export_onnx.py \\
        --checkpoint khanhld/chunkformer-rnnt-small-vie-stream-dct --streaming \\
        --chunk-size 64 --left-context 128 --right-context 8 --out-dir onnx/rnnt_stream

``--checkpoint`` accepts a local directory or a Hugging Face Hub repo id.
"""

import argparse
import json
import os
from typing import Any, Dict

import torch

from chunkformer.chunkformer_model import ChunkFormerModel
from chunkformer.onnx.wrappers import (
    CtcOnnx,
    EncoderChunkOnnx,
    EncoderFullOnnx,
    JointOnnx,
    PredictorStepOnnx,
)

OPSET = 17


def _maybe_simplify(path: str) -> None:
    """Run onnxsim on a saved graph if the package is available."""
    try:
        import onnx
        from onnxsim import simplify
    except ImportError:
        print("  (onnxsim not installed; skipping simplification)")
        return
    model = onnx.load(path)
    model_simp, ok = simplify(model)
    if ok:
        onnx.save(model_simp, path)
        print("  simplified")
    else:
        print("  simplification failed; keeping original")


def export_encoder_full(
    encoder: torch.nn.Module,
    feat_dim: int,
    out_dir: str,
    simplify: bool,
    chunk_size: int = 0,
    left: int = 0,
    right: int = 0,
):
    wrapper = EncoderFullOnnx(encoder, chunk_size, left, right).eval()
    feats = torch.randn(1, 400, feat_dim)
    feat_lens = torch.tensor([400], dtype=torch.int64)
    path = os.path.join(out_dir, "encoder_full.onnx")
    torch.onnx.export(
        wrapper,
        (feats, feat_lens),
        path,
        input_names=["feats", "feat_lens"],
        output_names=["enc_out", "enc_lens"],
        dynamic_axes={
            "feats": {0: "batch", 1: "time"},
            "feat_lens": {0: "batch"},
            "enc_out": {0: "batch", 1: "out_time"},
            "enc_lens": {0: "batch"},
        },
        opset_version=OPSET,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"  wrote {path}")
    if simplify:
        _maybe_simplify(path)


def export_encoder_chunk(
    encoder: Any,
    feat_dim: int,
    chunk_size: int,
    left: int,
    right: int,
    out_dir: str,
    simplify: bool,
) -> Dict[str, int]:
    sub = encoder.embed.subsampling_rate
    heads = encoder.attention_heads
    num_blocks = encoder.num_blocks
    d_out = encoder.output_size()
    conv_lorder = encoder.cnn_module_kernel // 2

    size = encoder.embed.reverse_calc_length(chunk_size) + right * sub
    stride = chunk_size * sub

    wrapper = EncoderChunkOnnx(encoder, chunk_size, left, right).eval()
    chunk = torch.randn(1, size, feat_dim)
    att_cache = torch.zeros(num_blocks, 1, heads, left, d_out // heads * 2)
    cnn_cache = torch.zeros(num_blocks, 1, d_out, conv_lorder)
    offset = torch.zeros((), dtype=torch.int64)

    path = os.path.join(out_dir, "encoder_chunk.onnx")
    torch.onnx.export(
        wrapper,
        (chunk, att_cache, cnn_cache, offset),
        path,
        input_names=["chunk", "att_cache", "cnn_cache", "offset"],
        output_names=["enc_out", "r_att_cache", "r_cnn_cache"],
        dynamic_axes={
            "chunk": {0: "batch", 1: "chunk_time"},
            "att_cache": {1: "batch"},
            "cnn_cache": {1: "batch"},
            "enc_out": {0: "batch", 1: "out_time"},
            "r_att_cache": {1: "batch"},
            "r_cnn_cache": {1: "batch"},
        },
        opset_version=OPSET,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"  wrote {path}")
    if simplify:
        _maybe_simplify(path)
    return {"size": int(size), "stride": int(stride)}


def export_ctc(ctc: torch.nn.Module, d_out: int, out_dir: str, simplify: bool):
    wrapper = CtcOnnx(ctc).eval()
    enc_out = torch.randn(1, 50, d_out)
    path = os.path.join(out_dir, "ctc.onnx")
    torch.onnx.export(
        wrapper,
        (enc_out,),
        path,
        input_names=["enc_out"],
        output_names=["log_probs"],
        dynamic_axes={
            "enc_out": {0: "batch", 1: "time"},
            "log_probs": {0: "batch", 1: "time"},
        },
        opset_version=OPSET,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"  wrote {path}")
    if simplify:
        _maybe_simplify(path)


def export_predictor(predictor: Any, out_dir: str, simplify: bool):
    wrapper = PredictorStepOnnx(predictor).eval()
    n_layers = predictor.n_layers
    hidden = predictor.hidden_size
    token = torch.zeros(1, 1, dtype=torch.int64)
    state_m = torch.zeros(n_layers, 1, hidden)
    state_c = torch.zeros(n_layers, 1, hidden)
    path = os.path.join(out_dir, "predictor.onnx")
    torch.onnx.export(
        wrapper,
        (token, state_m, state_c),
        path,
        input_names=["token", "state_m", "state_c"],
        output_names=["pred_out", "new_m", "new_c"],
        dynamic_axes={
            "token": {0: "batch"},
            "state_m": {1: "batch"},
            "state_c": {1: "batch"},
            "pred_out": {0: "batch"},
            "new_m": {1: "batch"},
            "new_c": {1: "batch"},
        },
        opset_version=OPSET,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"  wrote {path}")
    if simplify:
        _maybe_simplify(path)


def export_joint(joint: torch.nn.Module, enc_dim: int, pred_dim: int, out_dir: str, simplify: bool):
    wrapper = JointOnnx(joint).eval()
    enc_t = torch.randn(1, 1, enc_dim)
    pred = torch.randn(1, 1, pred_dim)
    path = os.path.join(out_dir, "joint.onnx")
    torch.onnx.export(
        wrapper,
        (enc_t, pred),
        path,
        input_names=["enc_t", "pred"],
        output_names=["logits"],
        dynamic_axes={
            "enc_t": {0: "batch"},
            "pred": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=OPSET,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"  wrote {path}")
    if simplify:
        _maybe_simplify(path)


@torch.no_grad()
def main():
    global OPSET
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Local dir or HF repo id of the ChunkFormer model"
    )
    parser.add_argument("--out-dir", required=True, help="Output directory for the ONNX graphs")
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Also export the cache-aware streaming chunk encoder",
    )
    parser.add_argument("--chunk-size", type=int, default=64, help="Streaming chunk size (frames)")
    parser.add_argument("--left-context", type=int, default=128, help="Streaming left context")
    parser.add_argument("--right-context", type=int, default=8, help="Streaming right context")
    parser.add_argument(
        "--full-chunk-size",
        type=int,
        default=0,
        help="Limited-context chunk size for the non-streaming encoder (0 = full context)",
    )
    parser.add_argument(
        "--full-left-context", type=int, default=0, help="Non-streaming limited left context"
    )
    parser.add_argument(
        "--full-right-context", type=int, default=0, help="Non-streaming limited right context"
    )
    parser.add_argument("--opset", type=int, default=OPSET, help="ONNX opset version")
    parser.add_argument("--simplify", action="store_true", help="Run onnxsim on each graph")
    args = parser.parse_args()

    OPSET = args.opset

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading checkpoint: {args.checkpoint}")
    hf_model = ChunkFormerModel.from_pretrained(args.checkpoint)
    hf_model.eval()
    inner = hf_model.model
    model_type = getattr(hf_model.config, "model", "asr_model")
    encoder = inner.encoder
    feat_dim = encoder.input_size
    d_out = encoder.output_size()
    has_ctc = getattr(inner, "ctc", None) is not None
    is_transducer = model_type == "transducer"

    vocab_size = (
        len(hf_model.char_dict) if hf_model.char_dict else getattr(hf_model.config, "output_dim", 0)
    )
    blank_id = int(getattr(inner, "blank", 0)) if is_transducer else 0

    # Feature-extraction config so the ONNX runtime can go from waveform to
    # fbank without loading the PyTorch model (CMVN is baked into the graph).
    fbank_conf = getattr(hf_model.config, "fbank_conf", {}) or {}
    resample_conf = getattr(hf_model.config, "resample_conf", {}) or {}

    meta: Dict[str, object] = {
        "model_type": model_type,
        "feat_dim": int(feat_dim),
        "encoder_output_size": int(d_out),
        "attention_heads": int(encoder.attention_heads),
        "num_blocks": int(encoder.num_blocks),
        "subsampling_rate": int(encoder.embed.subsampling_rate),
        "cnn_module_kernel": int(encoder.cnn_module_kernel),
        "cnn_lorder": int(encoder.cnn_module_kernel // 2),
        "vocab_size": int(vocab_size),
        "blank_id": int(blank_id),
        "has_ctc": bool(has_ctc),
        "is_transducer": bool(is_transducer),
        "streaming": bool(args.streaming),
        "opset": int(args.opset),
        "feature": {
            "sample_rate": int(resample_conf.get("resample_rate", 16000)),
            "num_mel_bins": int(fbank_conf.get("num_mel_bins", feat_dim)),
            "frame_length": int(fbank_conf.get("frame_length", 25)),
            "frame_shift": int(fbank_conf.get("frame_shift", 10)),
        },
    }

    # Dump the vocabulary so text output needs no external files.
    if hf_model.char_dict:
        vocab_path = os.path.join(args.out_dir, "vocab.txt")
        with open(vocab_path, "w", encoding="utf-8") as f:
            for idx in sorted(hf_model.char_dict):
                f.write(f"{hf_model.char_dict[idx]} {idx}\n")
        print(f"  wrote {vocab_path}")

    if args.full_chunk_size > 0:
        print(
            f"Exporting non-streaming encoder (limited context "
            f"chunk={args.full_chunk_size} left={args.full_left_context} "
            f"right={args.full_right_context})..."
        )
        meta["full_context"] = {
            "chunk_size": int(args.full_chunk_size),
            "left_context_size": int(args.full_left_context),
            "right_context_size": int(args.full_right_context),
        }
    else:
        print("Exporting full-context encoder...")
    export_encoder_full(
        encoder,
        feat_dim,
        args.out_dir,
        args.simplify,
        chunk_size=args.full_chunk_size,
        left=args.full_left_context,
        right=args.full_right_context,
    )

    if args.streaming:
        print("Exporting streaming chunk encoder...")
        stream_meta = export_encoder_chunk(
            encoder,
            feat_dim,
            args.chunk_size,
            args.left_context,
            args.right_context,
            args.out_dir,
            args.simplify,
        )
        meta["stream"] = {
            "chunk_size": int(args.chunk_size),
            "left_context_size": int(args.left_context),
            "right_context_size": int(args.right_context),
            **stream_meta,
        }

    if has_ctc:
        print("Exporting CTC head...")
        export_ctc(inner.ctc, d_out, args.out_dir, args.simplify)

    if is_transducer:
        print("Exporting RNN-T predictor...")
        export_predictor(inner.predictor, args.out_dir, args.simplify)
        print("Exporting RNN-T joint...")
        export_joint(inner.joint, d_out, inner.predictor.output_size(), args.out_dir, args.simplify)
        meta["predictor"] = {
            "n_layers": int(inner.predictor.n_layers),
            "hidden_size": int(inner.predictor.hidden_size),
            "output_size": int(inner.predictor.output_size()),
        }

    meta_path = os.path.join(args.out_dir, "onnx_config.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote metadata: {meta_path}")
    print("Done.")


if __name__ == "__main__":
    main()
