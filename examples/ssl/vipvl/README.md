# ViP-VL Self-Supervised Pretraining (ChunkFormer)

This recipe pretrains a ChunkFormer **encoder** with **ViP-VL** (Vietnamese
self-supervised speech Pretraining via Vector-quantization Learning) and exports
a finetune-ready encoder checkpoint. ViP-VL builds on the random-projection
quantizer technique of [BEST-RQ](https://arxiv.org/abs/2202.01855).

ViP-VL masks the input fbank features and trains the encoder to predict the
codebook ids produced by a *frozen* random-projection quantizer over the
unmasked features. There are **no text labels**, so this recipe has no
tokenizer / BPE / decoding / WER stages.

## Pipeline (`run.sh`)

| Stage | Description |
|------|-------------|
| 0 | Data format conversion (`data.tsv` → shard/raw list) |
| 1 | Global CMVN computation |
| 2 | ViP-VL self-supervised pretraining (DDP) |
| 3 | Average checkpoints + export an **encoder-only** checkpoint |
| 4 | (optional) Push the encoder checkpoint to the Hugging Face Hub |

Run a range of stages with:

```bash
bash run.sh --stage 2 --stop_stage 3
```

## Data layout

```
data/
  train/
    data.tsv        # columns include the audio path (wav)
    wav.scp         # produced by stage 0 (tools/tsv_to_list.py)
    shards.list     # if data_type=shard
  dev/
    ...
```

## Config

`conf/vipvl.yaml` is the configuration used to train the released checkpoint
(`avg_50.pt`). Key parts:

- `model: vipvl` with `model_conf` (codebook `latent_vars`, `latent_groups`,
  `latent_dim`, masking `mask_prob` / `mask_length`, `dist_fn`).
- `dataset: ssl` with `crop_conf.crop_length` to bound sequence length.
- `encoder_conf.encoder_layerdrop` enables LayerDrop during pretraining.

> The dataset paths in the config are placeholders (`data/train/...`). Point them
> at your own corpus; no private data location is shipped.

## Exported encoder checkpoint (stage 3)

Stage 3 calls `tools/export_ssl_encoder.py`, which keeps only the `encoder.*`
weights (dropping the ViP-VL quantizer / prediction head / mask embedding) and
writes a self-contained, **sanitised** bundle:

```
exp/vipvl/encoder_checkpoint/
  pytorch_model.pt   # encoder-only weights
  config.yaml        # encoder_conf + feature config (no data paths)
  global_cmvn        # CMVN statistics
  README.md
```

## Finetuning on a downstream task

The encoder weights load via `strict=False`, so point any ASR / RNN-T /
classification recipe at the checkpoint and train the task heads from scratch.
Make sure the downstream `encoder_conf` matches `config.yaml`.

The `checkpoint` (and `enc_init`) argument accepts **either a local path or a
Hugging Face Hub repo id** — `load_checkpoint` first looks for a local file or a
local directory containing `pytorch_model.pt`, and otherwise downloads
`pytorch_model.pt` from the Hub (cached locally):

```bash
# In examples/asr/ctc/run.sh (or rnnt / classification)

# Option A — local exported bundle (file or directory)
checkpoint=/path/to/exp/vipvl/encoder_checkpoint/pytorch_model.pt

# Option B — download straight from the Hugging Face Hub
checkpoint=khanhld/vip-vl-base-vie
```

> For a **private** Hub repo, authenticate first via `huggingface-cli login` or
> by exporting `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`). Public repos need no token.
