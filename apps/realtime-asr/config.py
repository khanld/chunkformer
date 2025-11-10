"""
Configuration for real-time streaming ASR application.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class StreamingConfig:
    """Configuration for streaming ASR"""

    # Model configuration
    model_path: str

    chunk_size: int = 6  # Chunk size in frames (post-subsampling) - middle value
    left_context_size: int = 50  # Number of left context frames - middle value
    right_context_size: int = 0  # Number of right context frames (streaming mode, no look-ahead)

    # Subsampling configuration
    subsampling_rate: int = 8  # Subsampling factor (8x for ChunkFormer)
    frame_shift_ms: float = 10.0  # Frame shift in ms (standard fbank)

    # Audio configuration
    sample_rate: int = 16000  # 16kHz audio
    device_index: Optional[int] = None  # Microphone device index

    # Model inference configuration
    device: str = "cuda"  # "cuda" or "cpu"

    # Feature extraction
    num_mel_bins: int = 80  # Number of mel filterbanks
    frame_length_ms: float = 25.0  # Frame length in ms

    def __post_init__(self):
        """Validate configuration and calculate derived values"""
        assert self.chunk_size > 0, "Chunk size must be positive"
        assert self.left_context_size >= 0, "Left context size must be non-negative"
        assert self.right_context_size >= 0, "Right context size must be non-negative"
        assert self.sample_rate in [
            8000,
            16000,
            22050,
            44100,
            48000,
        ], "Sample rate must be standard audio rate"
        assert self.subsampling_rate > 0, "Subsampling rate must be positive"

        # Calculate chunk duration in ms
        # chunk_size (frames after subsampling) -> frames before subsampling -> ms
        self.chunk_duration_ms = self.chunk_size * self.subsampling_rate * self.frame_shift_ms

    def get_algorithmic_latency_ms(self) -> float:
        """Calculate algorithmic latency in milliseconds"""
        latency: float = self.chunk_duration_ms + (self.right_context_size * self.frame_shift_ms)
        return latency

    def get_chunk_size_samples(self) -> int:
        """Get chunk size in audio samples"""
        return int(self.sample_rate * self.chunk_duration_ms / 1000)

    def get_chunk_size_features(self) -> int:
        """Get chunk size in feature frames (before subsampling)"""
        return int(self.chunk_duration_ms / self.frame_shift_ms)

    def get_chunk_size_subsampled(self) -> int:
        """Get chunk size after subsampling (same as input chunk_size)"""
        return self.chunk_size

    def __repr__(self) -> str:
        """Pretty print configuration"""
        lines = [
            "StreamingConfig:",
            f"  Model: {self.model_path}",
            f"  Chunk size: {self.chunk_size} frames ({self.chunk_duration_ms:.0f}ms)",
            f"  Context: L={self.left_context_size}, R={self.right_context_size}",
            f"  Algorithmic latency: {self.get_algorithmic_latency_ms():.1f}ms",
            f"  Subsampling: {self.subsampling_rate}x",
            f"  Sample rate: {self.sample_rate}Hz",
            f"  Device: {self.device}",
        ]
        return "\n".join(lines)


# Preset configurations for different use cases
PRESETS = {
    "ultra_low_latency": StreamingConfig(
        model_path="",  # To be filled
        chunk_size=4,  # 320ms (4 * 8 * 10ms) - from training
        left_context_size=40,  # From training
        right_context_size=0,  # Streaming mode (from training)
    ),
    "low_latency": StreamingConfig(
        model_path="",
        chunk_size=6,  # 480ms (6 * 8 * 10ms) - from training (default)
        left_context_size=50,  # From training (default)
        right_context_size=0,  # Streaming mode (from training)
    ),
    "balanced": StreamingConfig(
        model_path="",
        chunk_size=8,  # 640ms (8 * 8 * 10ms) - from training
        left_context_size=60,  # From training
        right_context_size=0,  # Streaming mode (from training)
    ),
    "high_accuracy": StreamingConfig(
        model_path="",
        chunk_size=8,  # 640ms - same as balanced
        left_context_size=60,  # Maximum from training
        right_context_size=2,  # Small look-ahead for better accuracy (optional)
    ),
}


def get_preset(preset_name: str, model_path: str) -> StreamingConfig:
    """
    Get a preset configuration.

    Args:
        preset_name: One of ["ultra_low_latency", "low_latency", "balanced", "high_accuracy"]
        model_path: Path to model checkpoint

    Returns:
        StreamingConfig with preset values
    """
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. " f"Available: {list(PRESETS.keys())}")

    config = PRESETS[preset_name]
    config.model_path = model_path
    return config


if __name__ == "__main__":
    """Test configurations"""
    print("Available presets:\n")
    for name, config in PRESETS.items():
        config.model_path = "/path/to/model.pt"
        print(f"{name}:")
        print(config)
        print(f"  Chunk samples: {config.get_chunk_size_samples()}")
        print(f"  Feature frames: {config.get_chunk_size_features()}")
        print(f"  Subsampled: {config.get_chunk_size_subsampled()}")
        print()
