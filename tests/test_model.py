from chunkformer import ChunkFormerModel

# Option 1: Load a pre-trained model from Hugging Face or local directory
model = ChunkFormerModel.from_pretrained("khanhld/chunkformer-large-vie")

# For single long-form audio transcription
transcription = model.endless_decode(
    audio_path="data/common_voice_vi_23397238.wav",
    chunk_size=64,
    left_context_size=128, 
    right_context_size=128,
    total_batch_duration=14400,  # in seconds
    return_timestamps=True
)
print(transcription)