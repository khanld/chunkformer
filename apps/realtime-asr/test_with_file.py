#!/usr/bin/env python3
"""
Test streaming ASR with an audio file (simulates real-time streaming).
Useful for testing without microphone or for reproducible benchmarks.
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from audio_capture import AudioFileSimulator
from config import StreamingConfig
from stream_asr import RealtimeASR


def main():
    parser = argparse.ArgumentParser(description="Test streaming ASR with audio file")
    parser.add_argument(
        "--model", type=str, required=True, help="Path to ChunkFormer model checkpoint"
    )
    parser.add_argument(
        "--audio", type=str, required=True, help="Path to audio file (wav, mp3, etc.)"
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
        help="Left context size in frames. Default: 50 (from training [40, 50, 60])",
    )
    parser.add_argument(
        "--right-context-size",
        type=int,
        default=0,
        help="Right context size in frames. Default: 0 (streaming mode, from training)",
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
        "--realtime", action="store_true", help="Simulate real-time by adding delays between chunks"
    )
    parser.add_argument("--output", type=str, default=None, help="Save transcription to file")

    args = parser.parse_args()

    # Check audio file exists (but not model - it can be from HuggingFace)
    if not Path(args.audio).exists():
        print(f"Error: Audio file not found at {args.audio}")
        sys.exit(1)

    # Create config
    config = StreamingConfig(
        model_path=args.model,
        chunk_size=args.chunk_size,
        left_context_size=args.left_context_size,
        right_context_size=args.right_context_size,
        device=args.device,
        sample_rate=args.sample_rate,
        device_index=None,  # Not used for file
    )

    # Create ASR engine (but don't start microphone)
    print("Initializing ASR engine...")
    asr = RealtimeASR(config)

    # Replace audio capture with file simulator
    asr.audio_capture = AudioFileSimulator(
        audio_path=args.audio,
        sample_rate=config.sample_rate,
        chunk_duration_ms=config.chunk_duration_ms,
        realtime=args.realtime,
    )

    print("\n" + "=" * 60)
    print(f"🎵 Processing audio file: {args.audio}")
    if args.realtime:
        print("   (simulating real-time with delays)")
    print(f"   Chunk size: {config.chunk_size} frames ({config.chunk_duration_ms:.0f}ms)")
    print("=" * 60)
    print()

    # Process file
    try:
        asr.audio_capture.start()

        chunk_count = 0
        start_time = time.time()
        transcription = []

        while True:
            # Get audio chunk from file
            audio_chunk = asr.audio_capture.read_chunk()

            if audio_chunk is None:
                break

            chunk_count += 1
            chunk_start = time.time()

            # Process chunk
            encoder_out, text = asr.process_chunk(audio_chunk)

            chunk_time = time.time() - chunk_start
            rtf = chunk_time / asr.chunk_duration_sec  # Real-time factor

            # Display results
            elapsed = time.time() - start_time
            if text.strip():
                transcription.append(text)
                print(f"[{elapsed:6.1f}s | RTF: {rtf:.3f}] {' '.join(transcription)}")

            # Show progress every 10 chunks
            if chunk_count % 10 == 0:
                print(
                    f"  → Processed {chunk_count} chunks, " f"{asr.total_frames_processed} frames"
                )

        # Final statistics
        total_time = time.time() - start_time
        audio_duration = chunk_count * config.chunk_duration_ms / 1000.0

        print("\n" + "=" * 60)
        print("✅ Processing complete")
        print("=" * 60)
        print(f"Chunks processed: {chunk_count}")
        print(f"Audio duration: {audio_duration:.2f}s")
        print(f"Processing time: {total_time:.2f}s")
        print(f"Average RTF: {total_time / audio_duration:.3f}")
        print(f"Speed: {audio_duration / total_time:.2f}x real-time")

        # Full transcription
        full_text = " ".join(transcription)
        print("\nFull transcription:")
        print(f"  {full_text}")

        # Save to file if requested
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(full_text + "\n")
            print(f"\n✓ Saved transcription to: {args.output}")

    except KeyboardInterrupt:
        print("\n\n🛑 Processing interrupted by user")

    finally:
        asr.audio_capture.stop()


if __name__ == "__main__":
    main()
