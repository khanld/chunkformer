#!/bin/bash

# Speech Classification Training Pipeline
# Stages:
# Stage 0: Data Format Conversion (TSV to list format)
# Stage 1: Feature Generation (CMVN computation)
# Stage 2: Label Statistics and Validation
# Stage 3: Training
# Stage 4: Evaluation
# Stage 5: Export Model for Inference
# Stage 6: Push Model to Hugging Face Hub (optional)

. ./path.sh || exit 1;

# GPU Configuration
export CUDA_VISIBLE_DEVICES="0"
echo "CUDA_VISIBLE_DEVICES is ${CUDA_VISIBLE_DEVICES}"

# Stage control
stage=0
stop_stage=6

# Multi-machine training settings
HOST_NODE_ADDR="localhost:0"
num_nodes=1
job_id=2024

# Data directory
wave_data=data
data_type=raw

# Training configuration
# Choose one of:
# - conf/multi_task.yaml: Single-task gender classification
# - conf/multi_task.yaml: Multi-task (gender + emotion + region)
train_config=conf/multi_task.yaml

# Training settings
checkpoint=
num_workers=4
dir=exp/multi_task
tensorboard_dir=tensorboard

# Model averaging
average_checkpoint=true
decode_checkpoint=$dir/final.pt
average_num=10

# Hugging Face Hub upload settings (optional)
# To enable upload and set these variables:
hf_token="hf_xxxxxxxxxxxxxxxxxxxxxxxxx"  # Your Hugging Face token
hf_repo_id="username/chunkformer-model"   # Your repository ID

# Dataset names (folder names under data/)
train_set=train
dev_set=dev
test_set=test

# Training engine
train_engine=torch_ddp

set -e
set -u
set -o pipefail

. tools/parse_options.sh || exit 1;

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "============================================"
    echo "Stage 0: Data Format Conversion"
    echo "============================================"

    # Convert data.tsv files to required format for training
    for dataset in $train_set $dev_set $test_set; do
        if [ -f "$wave_data/$dataset/data.tsv" ]; then
            echo "Converting $wave_data/$dataset/data.tsv"
            python tools/tsv_to_list.py $wave_data/$dataset/data.tsv
        else
            echo "Warning: $wave_data/$dataset/data.tsv not found, skipping..."
        fi
    done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "============================================"
    echo "Stage 1: Feature Generation (CMVN)"
    echo "============================================"

    tools/compute_cmvn_stats.py \
        --num_workers 16 \
        --train_config $train_config \
        --in_scp $wave_data/$train_set/wav.scp \
        --out_cmvn $wave_data/$train_set/global_cmvn
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "============================================"
    echo "Stage 2: Label Statistics and Validation"
    echo "============================================"

    python tools/compute_label_stats.py \
        --config $train_config \
        --train_data $wave_data/$train_set/data.list \
        --dev_data $wave_data/$dev_set/data.list \
        --test_data $wave_data/$test_set/data.list \
        --output_dir $wave_data
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "============================================"
    echo "Stage 3: Model Training"
    echo "============================================"

    mkdir -p $dir
    num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
    dist_backend="nccl"

    echo "Number of nodes: $num_nodes, GPUs per node: $num_gpus"

    torchrun --nnodes=$num_nodes --nproc_per_node=$num_gpus \
             --rdzv_endpoint=$HOST_NODE_ADDR \
             --rdzv_id=$job_id --rdzv_backend="c10d" \
        ${CHUNKFORMER_DIR}/chunkformer/bin/train.py \
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
    echo "============================================"
    echo "Stage 4: Model Evaluation"
    echo "============================================"

    mkdir -p $dir/test

    # Model averaging
    if [ ${average_checkpoint} == true ]; then
        decode_checkpoint=$dir/avg_${average_num}.pt
        echo "Averaging last $average_num checkpoints -> $decode_checkpoint"
        python ${CHUNKFORMER_DIR}/chunkformer/bin/average_model.py \
            --dst_model $decode_checkpoint \
            --src_path $dir \
            --num ${average_num}
    fi

    # Chunking settings (optional)
    chunk_size=
    left_context_size=
    right_context_size=

    # Evaluate on test set
    for test in $test_set; do
        result_dir=$dir/${test}
        mkdir -p $result_dir

        echo "Evaluating on $test set..."
        python ${CHUNKFORMER_DIR}/chunkformer/bin/classify.py \
            --gpu 0 \
            --config $dir/train.yaml \
            --data_type raw \
            --test_data $wave_data/$test/data.list \
            --checkpoint $decode_checkpoint \
            --batch_size 32 \
            --result_dir $result_dir \
            ${chunk_size:+--chunk_size $chunk_size} \
            ${left_context_size:+--left_context_size $left_context_size} \
            ${right_context_size:+--right_context_size $right_context_size}

        # Compute metrics
        python tools/compute_classification_metrics.py \
            --config $dir/train.yaml \
            --predictions $result_dir/predictions.tsv \
            --labels $wave_data/$test/data.list \
            --output $result_dir/metrics.txt

        echo "Results saved to $result_dir"
        echo "Metrics:"
        cat $result_dir/metrics.txt
    done
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "============================================"
    echo "Stage 5: Export Model for Inference"
    echo "============================================"

    # Recreate average checkpoint if needed
    if [ ${average_checkpoint} == true ]; then
        decode_checkpoint=$dir/avg_${average_num}.pt
        if [ ! -f "$decode_checkpoint" ]; then
            echo "Creating averaged checkpoint..."
            python ${CHUNKFORMER_DIR}/chunkformer/bin/average_model.py \
                --dst_model $decode_checkpoint \
                --src_path $dir \
                --num ${average_num}
        fi
        checkpoint_name="avg_${average_num}"
    else
        decode_checkpoint=$dir/final.pt
        checkpoint_name="final"
    fi

    # Create inference model directory
    inference_model_dir=$dir/model_checkpoint_${checkpoint_name}
    mkdir -p $inference_model_dir

    echo "Creating inference model directory: $inference_model_dir"

    # Copy model checkpoint
    if [ -f "$decode_checkpoint" ]; then
        cp $decode_checkpoint $inference_model_dir/pytorch_model.pt
        echo "✓ Copied model checkpoint"
    else
        echo "✗ Warning: Model checkpoint not found at $decode_checkpoint"
    fi

    # Copy training configuration
    if [ -f "$dir/train.yaml" ]; then
        cp $dir/train.yaml $inference_model_dir/config.yaml
        echo "✓ Copied training config"
    else
        echo "✗ Warning: Training config not found"
    fi

    # Copy CMVN statistics
    if [ -f "$wave_data/$train_set/global_cmvn" ]; then
        cp $wave_data/$train_set/global_cmvn $inference_model_dir/global_cmvn
        echo "✓ Copied CMVN stats"
    else
        echo "✗ Warning: CMVN statistics not found"
    fi

    # Copy label mapping JSON
    if [ -f "$wave_data/$train_set/label_mapping.json" ]; then
        cp $wave_data/$train_set/label_mapping.json $inference_model_dir/label_mapping.json
        echo "✓ Copied label_mapping.json"
    else
        echo "✗ Warning: label_mapping.json not found in $wave_data/$train_set"
    fi

    echo ""
    echo "============================================"
    echo "Model Export Complete!"
    echo "============================================"
    echo "Model directory: $inference_model_dir"
    echo ""
    echo "Directory contents:"
    ls -lh $inference_model_dir
    echo ""
    echo "You can now use this model for inference:"
    echo "  python chunkformer/bin/classify.py --checkpoint $inference_model_dir ..."
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "============================================"
    echo "Stage 6: Push Model to Hugging Face Hub"
    echo "============================================"

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
        echo ""
        echo "To enable upload, set the following variables in this script:"
        echo "  hf_token=\"your_huggingface_token\""
        echo "  hf_repo_id=\"username/repository-name\""
        echo ""
        echo "You can also upload manually later using:"
        echo "  cd ../../.."  # Go to chunkformer root
        echo "  python tools/push_model_hf.py \\"
        echo "    --model_dir $inference_model_dir \\"
        echo "    --repo_id username/repo-name \\"
        echo "    --token your_token"
    else
        echo "Uploading classification model to Hugging Face Hub..."
        echo "Repository: $hf_repo_id"
        echo "Model directory: $inference_model_dir"

        # Run the upload script
        python tools/push_model_hf.py \
            --model_dir "$inference_model_dir" \
            --repo_id "$hf_repo_id" \
            --token "$hf_token" \
            --commit_message "Upload ChunkFormer Classification Model"

        upload_status=$?

        if [ $upload_status -eq 0 ]; then
            echo ""
            echo "🎉 Classification model successfully uploaded to Hugging Face Hub!"
            echo "Model URL: https://huggingface.co/$hf_repo_id"
            echo ""
            echo "You can now load your model from anywhere with:"
            echo "from chunkformer import ChunkFormerModel"
            echo "model = ChunkFormerModel.from_pretrained('$hf_repo_id')"
            echo ""
            echo "Example usage:"
            echo "result = model.classify_audio("
            echo "    audio_path='path/to/audio.wav'"
            echo ")"
        else
            echo "❌ Failed to upload model to Hugging Face Hub"
            echo ""
            echo "You can try uploading manually with:"
            echo "  cd ../../.."
            echo "  python tools/push_model_hf.py \\"
            echo "    --model_dir $inference_model_dir \\"
            echo "    --repo_id $hf_repo_id \\"
            echo "    --token $hf_token"
        fi
    fi
fi

echo ""
echo "============================================"
echo "Training Pipeline Complete!"
echo "============================================"
