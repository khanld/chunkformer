"""
Audio capture utilities for real-time streaming ASR.
Handles microphone input with proper buffering and chunk management.
"""

import queue
from typing import Optional

import numpy as np
import sounddevice as sd


class AudioStreamCapture:
    """Captures audio from microphone in real-time with proper buffering"""

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_duration_ms: int = 480,
        device_index: Optional[int] = None,
        channels: int = 1,
        dtype: str = "float32",
    ):
        """
        Initialize audio stream capture.

        Args:
            sample_rate: Audio sample rate in Hz
            chunk_duration_ms: Duration of each chunk in milliseconds
            device_index: Specific device index to use (None for default)
            channels: Number of audio channels (1 for mono)
            dtype: Data type for audio samples
        """
        self.sample_rate = sample_rate
        self.chunk_duration_ms = chunk_duration_ms
        self.device_index = device_index
        self.channels = channels
        self.dtype = dtype

        # Calculate chunk size in samples
        self.chunk_size = int(sample_rate * chunk_duration_ms / 1000)

        # Audio buffer queue
        self.audio_queue: queue.Queue = queue.Queue()

        # Stream object
        self.stream = None
        self.is_running = False

        # Print available devices
        self._print_devices()

    def _print_devices(self):
        """Print available audio input devices"""
        print("\nAvailable audio devices:")
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device["max_input_channels"] > 0:
                marker = " *" if i == sd.default.device[0] else ""
                print(
                    f"  [{i}] {device['name']} "
                    f"(inputs: {device['max_input_channels']}, "
                    f"rate: {device['default_samplerate']}){marker}"
                )
        print()

    def _audio_callback(self, indata, frames, time_info, status):
        """Callback function called by sounddevice for each audio block"""
        if status:
            print(f"Audio callback status: {status}")

        # Copy audio data to queue
        # indata shape: (frames, channels)
        audio_data = indata.copy()

        # Convert to mono if needed
        if audio_data.shape[1] > 1:
            audio_data = audio_data.mean(axis=1, keepdims=True)

        # Flatten to 1D
        audio_data = audio_data.flatten()

        # Add to queue
        self.audio_queue.put(audio_data)

    def start(self):
        """Start audio capture stream"""
        if self.is_running:
            print("Audio stream already running")
            return

        try:
            # Create input stream
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.dtype,
                blocksize=self.chunk_size,
                device=self.device_index,
                callback=self._audio_callback,
            )

            self.stream.start()
            self.is_running = True

            device_info = sd.query_devices(self.device_index or sd.default.device[0])
            print(f"✓ Audio capture started on device: {device_info['name']}")
            print(f"  Sample rate: {self.sample_rate} Hz")
            print(f"  Chunk size: {self.chunk_size} samples ({self.chunk_duration_ms}ms)")
            print(f"  Channels: {self.channels}")

        except Exception as e:
            print(f"Error starting audio stream: {e}")
            raise

    def stop(self):
        """Stop audio capture stream"""
        if not self.is_running:
            return

        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        self.is_running = False
        print("✓ Audio capture stopped")

    def read_chunk(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        Read one audio chunk from the queue.

        Args:
            timeout: Maximum time to wait for chunk (seconds)

        Returns:
            Audio chunk as numpy array, or None if timeout
        """
        try:
            audio_chunk = self.audio_queue.get(timeout=timeout)
            return audio_chunk
        except queue.Empty:
            return None

    def clear_buffer(self):
        """Clear the audio buffer queue"""
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        print("Audio buffer cleared")

    def get_buffer_size(self) -> int:
        """Get current number of chunks in buffer"""
        return self.audio_queue.qsize()

    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()


class AudioFileSimulator:
    """Simulates streaming by reading from an audio file"""

    def __init__(
        self,
        audio_path: str,
        sample_rate: int = 16000,
        chunk_duration_ms: int = 480,
        realtime: bool = True,
    ):
        """
        Initialize file-based audio simulator.

        Args:
            audio_path: Path to audio file
            sample_rate: Target sample rate
            chunk_duration_ms: Duration of each chunk in milliseconds
            realtime: If True, simulate real-time by adding delays
        """
        import torchaudio

        self.audio_path = audio_path
        self.sample_rate = sample_rate
        self.chunk_duration_ms = chunk_duration_ms
        self.realtime = realtime

        # Load audio
        waveform, orig_sr = torchaudio.load(audio_path, normalize=False)

        # Resample if needed
        if orig_sr != sample_rate:
            resampler = torchaudio.transforms.Resample(orig_sr, sample_rate)
            waveform = resampler(waveform)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        self.audio = waveform.squeeze().numpy()
        self.chunk_size = int(sample_rate * chunk_duration_ms / 1000)
        self.position = 0
        self.is_running = False

        print(f"✓ Loaded audio file: {audio_path}")
        print(f"  Duration: {len(self.audio) / sample_rate:.2f}s")
        print(f"  Samples: {len(self.audio)}")

    def start(self):
        """Start simulation"""
        self.is_running = True
        self.position = 0
        print("✓ File simulation started")

    def stop(self):
        """Stop simulation"""
        self.is_running = False
        print("✓ File simulation stopped")

    def read_chunk(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Read next chunk from file"""
        if not self.is_running:
            return None

        if self.position >= len(self.audio):
            return None

        # Get chunk
        end_pos = min(self.position + self.chunk_size, len(self.audio))
        chunk = self.audio[self.position : end_pos]

        # Pad if last chunk is shorter
        if len(chunk) < self.chunk_size:
            chunk = np.pad(chunk, (0, self.chunk_size - len(chunk)), mode="constant")

        self.position = end_pos

        # Simulate real-time processing
        if self.realtime:
            import time

            time.sleep(self.chunk_duration_ms / 1000.0)

        return chunk

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


if __name__ == "__main__":
    """Test audio capture"""
    print("Testing audio capture for 5 seconds...")

    with AudioStreamCapture(sample_rate=16000, chunk_duration_ms=480) as capture:
        import time

        start = time.time()
        chunk_count = 0

        while time.time() - start < 5.0:
            chunk = capture.read_chunk()
            if chunk is not None:
                chunk_count += 1
                print(
                    f"Chunk {chunk_count}: {chunk.shape}, " f"RMS: {np.sqrt(np.mean(chunk**2)):.4f}"
                )

        print(f"\nCaptured {chunk_count} chunks in 5 seconds")
