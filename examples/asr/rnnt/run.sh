#!/bin/bash

# Training pipeline stages:
# Stage 0: Data Format Conversion
# Stage 1: Feature Generation
# Stage 2: Dictionary and Json Data Preparation
# Stage 3: Training
# Stage 4: Testing/Recognition
# Stage 5: Setting up model for ChunkFormer inference
# Stage 6: Push model to Hugging Face Hub (optional)

. ./path.sh || exit 1;

# You can also manually specify CUDA_VISIBLE_DEVICES
# if you don't want to utilize all available GPU resources.
export CUDA_VISIBLE_DEVICES="0"
echo "CUDA_VISIBLE_DEVICES is ${CUDA_VISIBLE_DEVICES}"
stage=0 # start from 0 if you need to start from data preparation
stop_stage=6

# You should change the following two parameters for multiple machine training,
# see https://pytorch.org/docs/stable/elastic/run.html
HOST_NODE_ADDR="localhost:0"
num_nodes=1
job_id=2023

# wav data dir
wave_data=data
data_type=raw
# Optional train_config
# 1. conf/train_transformer_large.yaml: Standard transformer
train_config=conf/v0.yaml
checkpoint=
num_workers=4

dir=exp/v0
tensorboard_dir=tensorboard

# use average_checkpoint will get better result
average_checkpoint=true
decode_checkpoint=$dir/final.pt
# maybe you can try to adjust it if you can not get close results as README.md
average_num=75
decode_modes="rnnt_greedy_search"

# bpemode (unigram or bpe)
nbpe=1024
bpemode=bpe

# Hugging Face Hub upload settings (optional)
# To enable upload and set these variables:
hf_token="hf_xxxxxxxxxxxxxxxxxxxxxxxxx"  # Your Hugging Face token
hf_repo_id="username/chunkformer-model"   # Your repository ID

set -e
set -u
set -o pipefail

train_set=train_set_name           # Your training folder name
dev_set=dev_set_name           # Your validation folder name
recog_set=test_set_name        # Your test folder name

train_engine=torch_ddp


. tools/parse_options.sh || exit 1;


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  ### Task dependent. Convert TSV data files to required format
  echo "stage 0: Data Format Conversion"

  # Convert data.tsv files to required format for training, dev, and test sets
  for dataset in $train_set $dev_set $recog_set; do
    if [ -f "$wave_data/$dataset/data.tsv" ]; then
      echo "Converting $wave_data/$dataset/data.tsv"
      python tools/tsv_to_list.py \
        $wave_data/$dataset/data.tsv
    else
      echo "Warning: $wave_data/$dataset/data.tsv not found, skipping..."
    fi
  done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  ### Task dependent. You have to design training and dev sets by yourself.
  ### But you can utilize Kaldi recipes in most cases
  echo "stage 1: Feature Generation"
  tools/compute_cmvn_stats.py --num_workers 16 --train_config $train_config \
    --in_scp $wave_data/$train_set/wav.scp \
    --out_cmvn $wave_data/$train_set/global_cmvn

fi


dict=$wave_data/lang_char/${train_set}_${bpemode}${nbpe}_units.txt
bpemodel=$wave_data/lang_char/${train_set}_${bpemode}${nbpe}
vocab=$wave_data/lang_char/${train_set}_${bpemode}${nbpe}.vocab
echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  ### Task dependent. You have to check non-linguistic symbols used in the corpus.
  echo "stage 2: Dictionary and Json Data Preparation"
  mkdir -p data/lang_char/

  echo "<blank> 0" > ${dict} # 0 will be used for "blank" in CTC
  echo "<unk> 1" >> ${dict} # <unk> must be 1
  echo "<sos/eos> 2" >> $dict # <eos>

  # we borrowed these code and scripts which are related bpe from ESPnet.
  cut -f 2- -d" " $wave_data/${train_set}/text > $wave_data/lang_char/input.txt
  tools/spm_train.py --input=$wave_data/lang_char/input.txt --vocab_size=${nbpe} --model_type=${bpemode} --model_prefix=${bpemodel} --input_sentence_size=100000000
  tools/spm_encode.py --model=${bpemodel}.model --output_format=piece < $wave_data/lang_char/input.txt | tr ' ' '\n' | sort | uniq | awk '{print $0 " " NR+2}' >> ${dict}
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  # Training
  mkdir -p $dir
  num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
  # Use "nccl" if it works, otherwise use "gloo"
  dist_backend="nccl"
  # train.py will write $train_config to $dir/train.yaml with model input
  # and output dimension, train.yaml will be used for inference or model
  # export later

  echo "$0: num_nodes is $num_nodes, proc_per_node is $num_gpus"
  torchrun --nnodes=$num_nodes --nproc_per_node=$num_gpus --rdzv_endpoint=$HOST_NODE_ADDR \
           --rdzv_id=$job_id --rdzv_backend="c10d" \
    chunkformer/bin/train.py \
      --use_amp \
      --train_engine ${train_engine} \
      --config $train_config \
      --data_type ${data_type} \
      --train_data $wave_data/$train_set/data.list \
      --cv_data $wave_data/$dev_set/data.list \
      ${checkpoint:+--checkpoint $checkpoint} \
      --model_dir $dir \
      --tensorboard_dir ${tensorboard_dir} \
      --ddp.dist_backend $dist_backend \
      --num_workers ${num_workers} \
      --pin_memory
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  # Test model, please specify the model you want to test by --checkpoint
  # TODO, Add model average here
  mkdir -p $dir/test
  if [ ${average_checkpoint} == true ]; then
    decode_checkpoint=$dir/avg_${average_num}.pt
    echo "do model average and final checkpoint is $decode_checkpoint"
    python chunkformer/bin/average_model.py \
      --dst_model $decode_checkpoint \
      --src_path $dir  \
      --num ${average_num}
  fi
  # Specify decoding_chunk_size if it's a unified dynamic chunk trained model
  # -1 for full chunk
  decoding_chunk_size=64
  num_decoding_left_chunks=128

  ctc_weight=0.3
  for test in $recog_set; do
    result_dir=$dir/${test}
    python chunkformer/bin/recognize.py --gpu 0 \
      --dtype fp16 \
      --modes $decode_modes \
      --config $dir/train.yaml \
      --data_type raw \
      --test_data $wave_data/$test/data.list \
      --checkpoint $decode_checkpoint \
      --beam_size 5 \
      --batch_size 16 \
      --blank_penalty 0.0 \
      --result_dir $result_dir \
      --ctc_weight $ctc_weight \
      ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size} \
      ${num_decoding_left_chunks:+--num_decoding_left_chunks $num_decoding_left_chunks}

    for mode in $decode_modes; do
      test_dir=$result_dir/$mode
      echo {$wave_data/$test/text}

      python tools/compute-wer.py --char=1 --v=1 \
        $wave_data/$test/text $test_dir/text > $test_dir/wer
    done
  done
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  # Setting Up Model for Inference
  echo "stage 5: Setting up model for ChunkFormer inference"
  # Step 0: Recreate average checkpoint if needed
  if [ ${average_checkpoint} == true ]; then
    decode_checkpoint=$dir/avg_${average_num}.pt
    echo "do model average and final checkpoint is $decode_checkpoint"
    python chunkformer/bin/average_model.py \
      --dst_model $decode_checkpoint \
      --src_path $dir  \
      --num ${average_num}
  fi


  # Step 1: Determine checkpoint and create appropriate directory name
  if [ ${average_checkpoint} == true ]; then
    source_checkpoint=$dir/avg_${average_num}.pt
    checkpoint_name="avg_${average_num}"
  else
    source_checkpoint=$dir/final.pt
    checkpoint_name="final"
  fi

  # Create inference model directory inside $dir
  inference_model_dir=$dir/model_checkpoint_${checkpoint_name}
  mkdir -p $inference_model_dir
  echo "Creating inference model directory: $inference_model_dir"

  if [ -f "$source_checkpoint" ]; then
    cp $source_checkpoint $inference_model_dir/pytorch_model.pt
    echo "Copied model checkpoint: $source_checkpoint -> $inference_model_dir/pytorch_model.pt"
  else
    echo "Warning: Model checkpoint not found at $source_checkpoint"
  fi

  # Step 2: Copy training configuration
  if [ -f "$dir/train.yaml" ]; then
    cp $dir/train.yaml $inference_model_dir/config.yaml
    echo "Copied training config: $dir/train.yaml -> $inference_model_dir/config.yaml"
  else
    echo "Warning: Training config not found at $dir/train.yaml"
  fi

  # Step 3: Copy CMVN statistics
  if [ -f "$wave_data/$train_set/global_cmvn" ]; then
    cp $wave_data/$train_set/global_cmvn $inference_model_dir/global_cmvn
    echo "Copied CMVN stats: $wave_data/$train_set/global_cmvn -> $inference_model_dir/global_cmvn"
  else
    echo "Warning: CMVN statistics not found at $wave_data/$train_set/global_cmvn"
  fi

  # Step 4: Copy vocabulary file
  if [ -f "$dict" ]; then
    cp $dict $inference_model_dir/vocab.txt
    echo "Copied vocabulary: $dict -> $inference_model_dir/vocab.txt"
  else
    echo "Warning: Vocabulary file not found at $dict"
  fi

  # Step 5: Copy tokenizer files (dict, bpemodel, vocab) to tokenizer folder
  tokenizer_dir=$inference_model_dir/tokenizer
  mkdir -p $tokenizer_dir
  if [ -f "$dict" ]; then
    cp $dict $tokenizer_dir/
    echo "Copied $dict to $tokenizer_dir/"
  else
    echo "Warning: $dict not found for tokenizer folder."
  fi
  if [ -f "${bpemodel}.model" ]; then
    cp ${bpemodel}.model $tokenizer_dir/
    echo "Copied ${bpemodel}.model to $tokenizer_dir/"
  else
    echo "Warning: ${bpemodel}.model not found for tokenizer folder."
  fi
  if [ -f "$vocab" ]; then
    cp $vocab $tokenizer_dir/
    echo "Copied $vocab to $tokenizer_dir/"
  else
    echo "Warning: $vocab not found for tokenizer folder."
  fi

  echo "Model setup completed. Directory structure:"
  ls -la $inference_model_dir
  echo ""
  echo "Your model is ready for ChunkFormer inference!"
  echo "Model directory: $inference_model_dir"
  echo ""
  echo "You can now use it with ChunkFormer:"
  echo "import chunkformer"
  echo "model = chunkformer.ChunkFormerModel.from_pretrained('$inference_model_dir')"
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  # Push model to Hugging Face Hub
  echo "stage 6: Pushing model to Hugging Face Hub"

  # Determine inference model directory (in case stage 5 was skipped)
  if [ ${average_checkpoint} == true ]; then
    checkpoint_name="avg_${average_num}"
  else
    checkpoint_name="final"
  fi
  inference_model_dir=$dir/model_checkpoint_${checkpoint_name}

  # Check if Hugging Face token and repo_id are provided
  if [ -z "$hf_token" ] || [ -z "$hf_repo_id" ]; then
    echo "Skipping Hugging Face upload: hf_token or hf_repo_id not provided"
    echo "To enable upload, set the following variables in this script:"
    echo "  hf_token=\"your_huggingface_token\""
    echo "  hf_repo_id=\"username/repository-name\""
    echo ""
    echo "You can also upload manually later using:"
    echo "  cd ../../.."  # Go to chunkformer root
    echo "  python tools/push_model_hf.py --model_dir $inference_model_dir --repo_id username/repo-name --token your_token"
  else
    echo "Uploading model to Hugging Face Hub..."
    echo "Repository: $hf_repo_id"
    echo "Model directory: $inference_model_dir"

    # Run the upload script
    python tools/push_model_hf.py \
      --model_dir "$inference_model_dir" \
      --repo_id "$hf_repo_id" \
      --token "$hf_token" \
      --commit_message "Upload ChunkFormer model"

    upload_status=$?

    if [ $upload_status -eq 0 ]; then
      echo ""
      echo "🎉 Model successfully uploaded to Hugging Face Hub!"
      echo "Model URL: https://huggingface.co/$hf_repo_id"
      echo ""
      echo "You can now load your model from anywhere with:"
      echo "from chunkformer import ChunkFormerModel"
      echo "model = ChunkFormerModel.from_pretrained('$hf_repo_id')"
    else
      echo "❌ Failed to upload model to Hugging Face Hub"
      echo "You can try uploading manually with:"
      echo "  cd ../../.."
      echo "  python tools/push_model_hf.py --model_dir $inference_model_dir --repo_id $hf_repo_id --token $hf_token"
    fi
  fi
fi
