export CUDA_VISIBLE_DEVICES="6"
python decode.py \
    --model_checkpoint "/raid/agi-ds/data-sharing/common/khanhle2/chunkformer/chunkformer-large-vie" \
    --total_batch_duration 1000 \
    --chunk_size 64 \
    --left_context_size 128 \
    --right_context_size 128 \
    --audio_list /raid/agi-ds/data-sharing/common/khanhle2/wenet/examples/librispeech/rnnt/data/test_vi/data.tsv
    # --long_form_audio /raid/agi-ds/data-sharing/common/khanhle2/chunkformer/data/common_voice_vi_23397238.wav

