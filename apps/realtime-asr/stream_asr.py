#!/usr/bin/env python3
"""
Real-time ASR Streaming Application
Continuously captures audio from microphone and decodes it using ChunkFormer
with 480ms chunk size for low-latency speech recognition.
"""

import argparse
import time
from typing import Tuple

import numpy as np
import torch
import torchaudio
from audio_capture import PYAUDIO_AVAILABLE, AudioStreamCapture, PyAudioStreamCapture
from config import StreamingConfig

from chunkformer import ChunkFormerModel
from chunkformer.utils.model_utils import get_output


class RealtimeASR:
    """Real-time ASR engine using ChunkFormer streaming decoding"""

    def __init__(self, config: StreamingConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Load model
        print(f"Loading model from {config.model_path}...")
        self.model = self._load_model()
        self.model.eval()
        self.model.to(self.device)

        # Calculate chunk parameters
        self.setup_chunk_parameters()

        self.audio_cache_frames = 7  # frames before subsampling
        self.audio_cache_duration_ms = self.audio_cache_frames * self.frame_shift_ms + 15  # 85ms
        self.audio_cache_samples = int(self.audio_cache_duration_ms * config.sample_rate / 1000)

        # Initialize caches (must be after audio_cache_samples is defined)
        self.att_cache = None
        self.cnn_cache = None
        self.offset = 0
        self.total_frames_processed = 0
        self.accumulated_text = ""  # Accumulate text across all chunks
        self.reset_cache()

        # Audio capture - prefer PyAudio on macOS for better stability
        # Use int16 format to match torchaudio.load(normalize=False)
        capture_class = PyAudioStreamCapture if PYAUDIO_AVAILABLE else AudioStreamCapture
        self.audio_capture = capture_class(
            sample_rate=config.sample_rate,
            chunk_duration_ms=config.chunk_duration_ms,
            device_index=config.device_index,
            dtype="int16",
        )

        print("✓ Model loaded successfully")
        print(f"✓ Chunk size: {config.chunk_size} frames ({config.chunk_duration_ms:.0f}ms)")
        print(f"✓ Left context: {config.left_context_size}")
        print(f"✓ Right context: {config.right_context_size}")
        print(
            f"✓ Audio cache: {self.audio_cache_frames} frames "
            f"({self.audio_cache_duration_ms:.0f}ms) for subsampling overlap"
        )

    def _load_model(self):
        """Load ChunkFormer model from checkpoint"""
        # Load model using ChunkFormerModel.from_pretrained()
        model = ChunkFormerModel.from_pretrained(self.config.model_path)

        return model

    def setup_chunk_parameters(self):
        """Calculate chunk parameters based on configuration"""
        # Chunk size is already in frames (post-subsampling) from config
        self.chunk_size = self.config.chunk_size

        # Get chunk duration in ms (already calculated in config)
        self.chunk_duration_ms = self.config.chunk_duration_ms
        self.chunk_duration_sec = self.chunk_duration_ms / 1000.0

        # Calculate chunk size in raw audio samples
        self.chunk_size_frames = self.config.get_chunk_size_samples()

        # Get feature extractor parameters
        self.frame_shift_ms = self.config.frame_shift_ms
        self.frame_shift_samples = int(self.frame_shift_ms * self.config.sample_rate / 1000)

        # Get chunk size in feature frames (before subsampling)
        self.chunk_size_features = self.config.get_chunk_size_features()

        # Subsampling rate
        self.subsampling_rate = self.config.subsampling_rate

        print("\nChunk parameters:")
        print(f"  - Chunk size: {self.chunk_size} frames (post-subsampling)")
        print(f"  - Duration: {self.chunk_duration_ms:.0f}ms")
        print(f"  - Audio samples: {self.chunk_size_frames}")
        print(f"  - Feature frames: {self.chunk_size_features} (pre-subsampling)")
        print(f"  - Subsampling: {self.subsampling_rate}x")

    def reset_cache(self):
        """Reset attention and convolution caches"""
        # Access encoder from the model (ChunkFormerModel wraps the actual model in self.model)
        encoder = self.model.model.encoder

        conv_lorder = encoder.cnn_module_kernel // 2

        self.att_cache = torch.zeros(
            (
                encoder.num_blocks,
                1,  # batch_size = 1 for streaming
                encoder.attention_heads,
                self.config.left_context_size,
                encoder._output_size // encoder.attention_heads * 2,
            ),
            dtype=torch.float32,
            device=self.device,
        )

        self.cnn_cache = torch.zeros(
            (encoder.num_blocks, 1, encoder._output_size, conv_lorder),
            dtype=torch.float32,
            device=self.device,
        )

        # Reset audio cache for subsampling overlap
        self.audio_cache = np.zeros(self.audio_cache_samples, dtype=np.float32)

        self.offset = 0
        self.total_frames_processed = 0
        self.accumulated_text = ""  # Reset accumulated text

    def extract_features(self, audio_chunk: np.ndarray) -> torch.Tensor:
        """Extract fbank features from audio chunk"""
        # Convert to torch tensor
        waveform = torch.from_numpy(audio_chunk).float().unsqueeze(0)  # [1, T]

        # Extract fbank features (80-dim)
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform,
            num_mel_bins=80,
            sample_frequency=self.config.sample_rate,
            frame_length=25.0,  # 25ms window
            frame_shift=10.0,  # 10ms shift
            energy_floor=0.0,
        )

        return fbank  # [T, 80]

    def process_chunk(self, audio_chunk: np.ndarray) -> Tuple[torch.Tensor, str]:
        """Process a single audio chunk through the model"""

        # Concatenate audio cache with new chunk
        audio_with_cache = np.concatenate([self.audio_cache, audio_chunk])

        # Extract features from the concatenated audio
        features = self.extract_features(audio_with_cache)  # [T, 80]
        features = features.unsqueeze(0).to(self.device)  # [1, T, 80]

        # Update audio cache: save last 7 frames worth of audio (85ms)
        self.audio_cache = audio_chunk[-self.audio_cache_samples :]

        # Forward through encoder with caching
        with torch.no_grad():
            encoder_out, _, self.att_cache, self.cnn_cache = self.model.model.encoder.forward_chunk(
                features,
                att_cache=self.att_cache,
                cnn_cache=self.cnn_cache,
                chunk_size=self.config.chunk_size,
                left_context_size=self.config.left_context_size,
                right_context_size=self.config.right_context_size,
                offset=self.offset,
            )

        # Decode (assuming CTC or similar)
        text = self.decode(encoder_out)

        # Update offset
        self.offset += self.chunk_size
        self.total_frames_processed += features.size(1)

        return encoder_out, text

    def decode(self, encoder_out: torch.Tensor) -> str:
        """Decode encoder output to text"""
        text: str
        if hasattr(self.model.model, "ctc"):
            # CTC decoding
            ctc_probs = self.model.model.ctc.log_softmax(encoder_out)  # [B, T, vocab]
            topk = ctc_probs.argmax(dim=-1)  # [B, T]
            hyps = [hyp.tolist() for hyp in topk]
            text = str(get_output(hyps, self.model.char_dict, self.model.config.model)[0])
        elif hasattr(self.model, "decoder"):
            # Transducer or attention decoder
            # Implement appropriate decoding here
            text = "[Decoder output]"
        else:
            text = "[Unknown decoder type]"

        return text

    def run(self):
        """Main streaming loop"""
        print("\n" + "=" * 60)
        print("🎤 Real-time ASR Streaming Started")
        print("=" * 60)
        print("Press Ctrl+C to stop\n")

        try:
            self.audio_capture.start()

            chunk_count = 0
            start_time = time.time()

            while True:
                # Get audio chunk from microphone
                audio_chunk = self.audio_capture.read_chunk()

                if audio_chunk is None:
                    continue

                chunk_count += 1
                chunk_start = time.time()

                # Process chunk
                encoder_out, text = self.process_chunk(audio_chunk)

                chunk_time = time.time() - chunk_start
                rtf = chunk_time / self.chunk_duration_sec  # Real-time factor

                # Accumulate text
                if text.strip():
                    self.accumulated_text += " " + text.strip()

                # Display results with accumulated text
                elapsed = time.time() - start_time
                if self.accumulated_text.strip():
                    print(
                        f"\r[{elapsed:6.1f}s | RTF: {rtf:.3f}] " f"{self.accumulated_text.strip()}",
                        end="",
                        flush=True,
                    )

                # Show progress every N chunks (on new line)
                if chunk_count % 10 == 0:
                    avg_rtf = (time.time() - start_time) / (chunk_count * self.chunk_duration_sec)
                    print(
                        f"\n  → Processed {chunk_count} chunks, "
                        f"{self.total_frames_processed} frames, "
                        f"avg RTF: {avg_rtf:.3f}"
                    )

        except KeyboardInterrupt:
            print("\n\n" + "=" * 60)
            print("🛑 Streaming stopped by user")
            print("=" * 60)
            total_time = time.time() - start_time
            print(f"Total chunks processed: {chunk_count}")
            print(f"Total time: {total_time:.2f}s")
            avg_rtf = total_time / (chunk_count * self.chunk_duration_sec)
            print(f"Average RTF: {avg_rtf:.3f}")

        finally:
            self.audio_capture.stop()


def main():
    parser = argparse.ArgumentParser(
        description="Real-time ASR with ChunkFormer streaming decoding"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Path to ChunkFormer model checkpoint"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=6,
        help=(
            "Chunk size in frames (post-subsampling). "
            "Default: 6 frames = 480ms. Training used [4, 6, 8]"
        ),
    )
    parser.add_argument(
        "--left-context-size",
        type=int,
        default=50,
        help=("Left context size in frames. " "Default: 50 (from training [40, 50, 60])"),
    )
    parser.add_argument(
        "--right-context-size",
        type=int,
        default=0,
        help=("Right context size in frames. " "Default: 0 (streaming mode, from training)"),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run model on (default: cuda if available)",
    )
    parser.add_argument(
        "--sample-rate", type=int, default=16000, help="Audio sample rate (default: 16000)"
    )
    parser.add_argument(
        "--mic-device",
        type=int,
        default=None,
        help="Microphone device index (default: prompt for selection)",
    )
    parser.add_argument(
        "--no-select-device",
        action="store_true",
        help="Skip interactive device selection and use default device",
    )

    args = parser.parse_args()

    # Always prompt for device selection unless --no-select-device is
    # specified or --mic-device is provided
    device_index = args.mic_device
    if device_index is None and not args.no_select_device:
        # Use the appropriate class for device selection
        if PYAUDIO_AVAILABLE:
            device_index = PyAudioStreamCapture.prompt_device_selection()
        else:
            device_index = AudioStreamCapture.prompt_device_selection()

    # Create config
    config = StreamingConfig(
        model_path=args.model,
        chunk_size=args.chunk_size,
        left_context_size=args.left_context_size,
        right_context_size=args.right_context_size,
        device=args.device,
        sample_rate=args.sample_rate,
        device_index=device_index,
    )

    # Create and run ASR engine
    asr = RealtimeASR(config)
    asr.run()


if __name__ == "__main__":
    main()
