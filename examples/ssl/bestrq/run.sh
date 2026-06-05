#!/bin/bash

# BEST-RQ self-supervised pretraining pipeline for the ChunkFormer encoder.
#
# Stages:
#   Stage 0: Data format conversion (TSV -> shard/raw list)
#   Stage 1: Global CMVN computation
#   Stage 2: Self-supervised pretraining (BEST-RQ)
#   Stage 3: Average checkpoints + export an encoder-only checkpoint for finetuning
#   Stage 4: Push the encoder checkpoint to the Hugging Face Hub (optional)
#
# Unlike the ASR recipes, BEST-RQ pretraining has NO text targets, so there is
# no tokenizer / BPE / decoding / WER stage.

. ./path.sh || exit 1;

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
echo "CUDA_VISIBLE_DEVICES is ${CUDA_VISIBLE_DEVICES}"

export OMP_NUM_THREADS=8

stage=2           # start from 0 for data preparation
stop_stage=3

# Distributed training rendezvous (see https://pytorch.org/docs/stable/elastic/run.html)
HOST_NODE_ADDR="localhost:0"
num_nodes=1
job_id=2023

# Data
wave_data=data
data_type=shard           # shard or raw
train_set=train           # training folder name under $wave_data
dev_set=dev               # validation folder name under $wave_data

# Training
train_config=conf/bestrq.yaml
checkpoint=
num_workers=4
dir=exp/bestrq
tensorboard_dir=tensorboard
train_engine=torch_ddp

# Checkpoint averaging / export
average_checkpoint=true
average_num=50
# Output directory of the finetune-ready encoder bundle
export_dir=$dir/encoder_checkpoint

# Hugging Face Hub upload settings (stage 4, optional)
hf_token=""                        # Your Hugging Face token
hf_repo_id=""                      # e.g. username/chunkformer-bestrq-encoder
hf_private=false

set -e
set -u
set -o pipefail

. tools/parse_options.sh || exit 1;


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "stage 0: Data Format Conversion"
  for dataset in $train_set $dev_set; do
    if [ -f "$wave_data/$dataset/data.tsv" ]; then
      echo "Converting $wave_data/$dataset/data.tsv"
      python tools/tsv_to_list.py $wave_data/$dataset/data.tsv
    else
      echo "Warning: $wave_data/$dataset/data.tsv not found, skipping..."
    fi
  done
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "stage 1: Global CMVN computation"
  tools/compute_cmvn_stats.py --num_workers 16 --train_config $train_config \
    --in_scp $wave_data/$train_set/wav.scp \
    --out_cmvn $wave_data/$train_set/global_cmvn
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "stage 2: BEST-RQ self-supervised pretraining"
  mkdir -p $dir
  num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
  dist_backend="nccl"   # use "gloo" if nccl is unavailable

  echo "$0: num_nodes is $num_nodes, proc_per_node is $num_gpus"
  torchrun --nnodes=$num_nodes --nproc_per_node=$num_gpus --rdzv_endpoint=$HOST_NODE_ADDR \
           --rdzv_id=$job_id --rdzv_backend="c10d" \
    ../../../chunkformer/bin/train.py \
      --use_amp \
      --train_engine ${train_engine} \
      --config $train_config \
      --data_type ${data_type} \
      --train_data $wave_data/$train_set/shards.list \
      --cv_data $wave_data/$dev_set/shards.list \
      ${checkpoint:+--checkpoint $checkpoint} \
      --model_dir $dir \
      --tensorboard_dir ${tensorboard_dir} \
      --ddp.dist_backend $dist_backend \
      --num_workers ${num_workers} \
      --pin_memory \
      --prefetch 2
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "stage 3: Average checkpoints + export encoder-only checkpoint"
  if [ ${average_checkpoint} == true ]; then
    decode_checkpoint=$dir/avg_${average_num}.pt
    echo "averaging the last ${average_num} checkpoints -> $decode_checkpoint"
    python ../../../chunkformer/bin/average_model.py \
      --dst_model $decode_checkpoint \
      --src_path $dir \
      --num ${average_num}
  else
    decode_checkpoint=$dir/final.pt
  fi

  # Strip the BEST-RQ heads (quantizer / final_proj / mask_emb) and keep only the
  # encoder.* weights so the checkpoint drops cleanly into ASR / RNN-T / classification
  # finetuning via --checkpoint (load_checkpoint uses strict=False).
  python ../../../tools/export_ssl_encoder.py \
    --checkpoint $decode_checkpoint \
    --config $dir/train.yaml \
    --cmvn $wave_data/$train_set/global_cmvn \
    --output_dir $export_dir

  echo "Encoder checkpoint exported to: $export_dir"
  echo "Finetune by pointing an ASR/RNN-T recipe's checkpoint= to $export_dir/pytorch_model.pt"
fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "stage 4: Push encoder checkpoint to Hugging Face Hub"
  if [ -z "$hf_token" ] || [ -z "$hf_repo_id" ]; then
    echo "Skipping upload: set hf_token and hf_repo_id to enable."
    echo "Manual upload:"
    echo "  python ../../../tools/push_model_hf.py --model_dir $export_dir --repo_id <user/repo> --token <token>"
  else
    private_flag=""
    if [ "$hf_private" == true ]; then private_flag="--private"; fi
    python ../../../tools/push_model_hf.py \
      --model_dir "$export_dir" \
      --repo_id "$hf_repo_id" \
      --token "$hf_token" \
      $private_flag \
      --commit_message "Upload ChunkFormer BEST-RQ pretrained encoder"
  fi
fi
