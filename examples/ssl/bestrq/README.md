# BEST-RQ Self-Supervised Pretraining (ChunkFormer)

This recipe pretrains a ChunkFormer **encoder** with
[BEST-RQ](https://arxiv.org/abs/2202.01855) (BERT-based Speech pre-Training with
Random-projection Quantizer) and exports a finetune-ready encoder checkpoint.

BEST-RQ masks the input fbank features and trains the encoder to predict the
codebook ids produced by a *frozen* random-projection quantizer over the
unmasked features. There are **no text labels**, so this recipe has no
tokenizer / BPE / decoding / WER stages.

## Pipeline (`run.sh`)

| Stage | Description |
|------|-------------|
| 0 | Data format conversion (`data.tsv` → shard/raw list) |
| 1 | Global CMVN computation |
| 2 | BEST-RQ self-supervised pretraining (DDP) |
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

`conf/bestrq.yaml` is the configuration used to train the released checkpoint
(`avg_50.pt`). Key parts:

- `model: bestrq` with `model_conf` (codebook `latent_vars`, `latent_groups`,
  `latent_dim`, masking `mask_prob` / `mask_length`, `dist_fn`).
- `dataset: ssl` with `crop_conf.crop_length` to bound sequence length.
- `encoder_conf.encoder_layerdrop` enables LayerDrop during pretraining.

> The dataset paths in the config are placeholders (`data/train/...`). Point them
> at your own corpus; no private data location is shipped.

## Exported encoder checkpoint (stage 3)

Stage 3 calls `tools/export_ssl_encoder.py`, which keeps only the `encoder.*`
weights (dropping the BEST-RQ quantizer / prediction head / mask embedding) and
writes a self-contained, **sanitised** bundle:

```
exp/bestrq/encoder_checkpoint/
  pytorch_model.pt   # encoder-only weights
  config.yaml        # encoder_conf + feature config (no data paths)
  global_cmvn        # CMVN statistics
  README.md
```

## Finetuning on a downstream task

The encoder weights load via `strict=False`, so point any ASR / RNN-T /
classification recipe at the exported checkpoint and train the task heads from
scratch. Make sure the downstream `encoder_conf` matches `config.yaml`.

```bash
# In examples/asr/ctc/run.sh (or rnnt / classification)
checkpoint=/path/to/exp/bestrq/encoder_checkpoint/pytorch_model.pt
```

## Reproducibility check

`tools/verify_bestrq_parity.py` runs a deterministic forward pass to confirm a
checkpoint reproduces expected BEST-RQ metrics:

```bash
python ../../../tools/verify_bestrq_parity.py \
    --config exp/bestrq/train.yaml \
    --checkpoint exp/bestrq/avg_50.pt
```
