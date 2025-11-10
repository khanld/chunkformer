"""
Audio capture utilities for real-time streaming ASR.
Handles microphone input with proper buffering and chunk management.
"""

import queue
from typing import Optional, Type, Union

import numpy as np

try:
    import pyaudio

    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False

try:
    import sounddevice as sd

    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False


class PyAudioStreamCapture:
    """Captures audio from microphone using PyAudio (more stable on macOS)"""

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_duration_ms: int = 480,
        device_index: Optional[int] = None,
        channels: int = 1,
        dtype: str = "int16",
        normalize: bool = False,
    ):
        """
        Initialize audio stream capture using PyAudio.

        Args:
            sample_rate: Audio sample rate in Hz
            chunk_duration_ms: Duration of each chunk in milliseconds
            device_index: Specific device index to use (None for default)
            channels: Number of audio channels (1 for mono)
            dtype: Data type for audio samples ('int16' or 'float32')
            normalize: If False, returns int16 audio like torchaudio.load(normalize=False)
        """
        if not PYAUDIO_AVAILABLE:
            raise ImportError("PyAudio is not installed. Install it with: pip install pyaudio")

        self.sample_rate = sample_rate
        self.chunk_duration_ms = chunk_duration_ms
        self.device_index = device_index
        self.channels = channels
        self.dtype = dtype
        self.normalize = normalize

        # Calculate chunk size in samples
        self.chunk_size = int(sample_rate * chunk_duration_ms / 1000)

        # Audio buffer queue
        self.audio_queue: queue.Queue = queue.Queue()

        # PyAudio objects
        self.pyaudio_instance = None
        self.stream = None
        self.is_running = False

        # Buffer for accumulating audio chunks to match desired chunk_size
        # Use int16 by default to match torchaudio.load(normalize=False)
        buffer_dtype = np.int16 if dtype == "int16" else np.float32
        self.audio_buffer = np.array([], dtype=buffer_dtype)

        # Print available devices
        self._print_devices()

    def _print_devices(self):
        """Print available audio input devices"""
        print("\nAvailable audio devices (PyAudio):")
        p = pyaudio.PyAudio()
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:
                marker = " *" if i == p.get_default_input_device_info()["index"] else ""
                print(
                    f"  [{i}] {info['name']} "
                    f"(inputs: {info['maxInputChannels']}, "
                    f"rate: {info['defaultSampleRate']}){marker}"
                )
        p.terminate()
        print()

    @staticmethod
    def prompt_device_selection() -> Optional[int]:
        """
        Prompt user to select an audio input device.

        Returns:
            Selected device index, or None for default device
        """
        p = pyaudio.PyAudio()

        # Get all input devices
        input_devices = []
        default_index = None

        try:
            default_info = p.get_default_input_device_info()
            default_index = default_info["index"]
        except Exception:
            pass

        print("\nAvailable audio input devices:")
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:
                input_devices.append(i)
                is_default = " (default)" if i == default_index else ""
                print(
                    f"  [{i}] {info['name']} - " f"{int(info['defaultSampleRate'])} Hz{is_default}"
                )

        p.terminate()

        if not input_devices:
            print("No input devices found!")
            return None

        # Prompt for selection
        device_list = ", ".join(map(str, input_devices))
        print(
            f"\nPress Enter for default device, or enter device number " f"[{device_list}]: ",
            end="",
        )

        try:
            user_input = input().strip()

            if user_input == "":
                # Use default
                return None

            device_index = int(user_input)

            if device_index in input_devices:
                return device_index
            else:
                print("Invalid device index. Using default device.")
                return None

        except (ValueError, KeyboardInterrupt):
            print("\nUsing default device.")
            return None

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback function called by PyAudio for each audio block"""
        if status:
            print(f"Audio callback status: {status}")

        # Convert bytes to numpy array based on dtype
        if self.dtype == "int16":
            audio_data = np.frombuffer(in_data, dtype=np.int16)
        else:
            audio_data = np.frombuffer(in_data, dtype=np.float32)

        # Accumulate audio data
        self.audio_buffer = np.concatenate([self.audio_buffer, audio_data])

        # Queue complete chunks of desired size
        while len(self.audio_buffer) >= self.chunk_size:
            chunk = self.audio_buffer[: self.chunk_size]
            self.audio_buffer = self.audio_buffer[self.chunk_size :]
            self.audio_queue.put(chunk)

        return (None, pyaudio.paContinue)

    def start(self):
        """Start audio capture stream"""
        if self.is_running:
            print("Audio stream already running")
            return

        try:
            self.pyaudio_instance = pyaudio.PyAudio()

            # Determine PyAudio format based on dtype
            if self.dtype == "int16":
                pa_format = pyaudio.paInt16
            else:
                pa_format = pyaudio.paFloat32

            # Open stream
            self.stream = self.pyaudio_instance.open(
                format=pa_format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback,
            )

            self.stream.start_stream()
            self.is_running = True

            device_index = (
                self.device_index
                if self.device_index is not None
                else self.pyaudio_instance.get_default_input_device_info()["index"]
            )
            device_info = self.pyaudio_instance.get_device_info_by_index(device_index)
            print(f"✓ Audio capture started on device: {device_info['name']}")
            print(f"  Sample rate: {self.sample_rate} Hz")
            print(f"  Chunk size: {self.chunk_size} samples ({self.chunk_duration_ms}ms)")
            print(f"  Channels: {self.channels}")
            print(f"  Data type: {self.dtype}")

        except Exception as e:
            print(f"Error starting audio stream: {e}")
            if self.pyaudio_instance:
                self.pyaudio_instance.terminate()
            raise

    def stop(self):
        """Stop audio capture stream"""
        if not self.is_running:
            return

        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

        if self.pyaudio_instance is not None:
            self.pyaudio_instance.terminate()
            self.pyaudio_instance = None

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


class AudioStreamCapture:
    """Captures audio from microphone in real-time with proper buffering"""

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_duration_ms: int = 480,
        device_index: Optional[int] = None,
        channels: int = 1,
        dtype: str = "int16",
        normalize: bool = False,
    ):
        """
        Initialize audio stream capture using sounddevice.

        Args:
            sample_rate: Audio sample rate in Hz
            chunk_duration_ms: Duration of each chunk in milliseconds
            device_index: Specific device index to use (None for default)
            channels: Number of audio channels (1 for mono)
            dtype: Data type for audio samples ('int16' or 'float32')
            normalize: If False, returns int16 audio like torchaudio.load(normalize=False)
        """
        if not SOUNDDEVICE_AVAILABLE:
            raise ImportError(
                "sounddevice is not installed. Install it with: pip install sounddevice"
            )

        self.sample_rate = sample_rate
        self.chunk_duration_ms = chunk_duration_ms
        self.device_index = device_index
        self.channels = channels
        self.dtype = dtype
        self.normalize = normalize

        # Calculate chunk size in samples
        self.chunk_size = int(sample_rate * chunk_duration_ms / 1000)

        # Audio buffer queue
        self.audio_queue: queue.Queue = queue.Queue()

        # Stream object
        self.stream = None
        self.is_running = False

        # Buffer for accumulating audio chunks to match desired chunk_size
        # Use int16 by default to match torchaudio.load(normalize=False)
        buffer_dtype = np.int16 if dtype == "int16" else np.float32
        self.audio_buffer = np.array([], dtype=buffer_dtype)

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

    @staticmethod
    def prompt_device_selection() -> Optional[int]:
        """
        Prompt user to select an audio input device.

        Returns:
            Selected device index, or None for default device
        """
        # Get all input devices
        devices = sd.query_devices()
        input_devices = []
        default_index = sd.default.device[0]

        print("\nAvailable audio input devices:")
        for i, device in enumerate(devices):
            if device["max_input_channels"] > 0:
                input_devices.append(i)
                is_default = " (default)" if i == default_index else ""
                print(
                    f"  [{i}] {device['name']} - "
                    f"{int(device['default_samplerate'])} Hz{is_default}"
                )

        if not input_devices:
            print("No input devices found!")
            return None

        # Prompt for selection
        device_list = ", ".join(map(str, input_devices))
        print(
            f"\nPress Enter for default device, or enter device number " f"[{device_list}]: ",
            end="",
        )

        try:
            user_input = input().strip()

            if user_input == "":
                # Use default
                return None

            device_index = int(user_input)

            if device_index in input_devices:
                return device_index
            else:
                print("Invalid device index. Using default device.")
                return None

        except (ValueError, KeyboardInterrupt):
            print("\nUsing default device.")
            return None

    def _audio_callback(self, indata, frames, time_info, status):
        """Callback function called by sounddevice for each audio block"""
        if status:
            print(f"Audio callback status: {status}")

        # Copy audio data
        # indata shape: (frames, channels)
        audio_data = indata.copy()

        # Convert to mono if needed
        if audio_data.shape[1] > 1:
            audio_data = audio_data.mean(axis=1, keepdims=True)

        # Flatten to 1D
        audio_data = audio_data.flatten()

        # Convert to the desired dtype if needed
        if self.dtype == "int16" and audio_data.dtype != np.int16:
            # sounddevice typically gives float32 in range [-1, 1]
            # Convert to int16 range [-32768, 32767]
            audio_data = (audio_data * 32767).astype(np.int16)
        elif self.dtype == "float32" and audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        # Accumulate audio data
        self.audio_buffer = np.concatenate([self.audio_buffer, audio_data])

        # Queue complete chunks of desired size
        while len(self.audio_buffer) >= self.chunk_size:
            chunk = self.audio_buffer[: self.chunk_size]
            self.audio_buffer = self.audio_buffer[self.chunk_size :]
            self.audio_queue.put(chunk)

    def start(self):
        """Start audio capture stream"""
        if self.is_running:
            print("Audio stream already running")
            return

        try:
            # On macOS, we need to use 0 for blocksize to let the system decide
            # This avoids the AUHAL component not found error
            blocksize = 0 if self.chunk_size else 0

            # Create input stream
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.dtype,
                blocksize=blocksize,  # Let system decide on macOS
                device=self.device_index,
                callback=self._audio_callback,
            )

            self.stream.start()
            self.is_running = True

            device_info = sd.query_devices(self.device_index or sd.default.device[0])
            print(f"✓ Audio capture started on device: {device_info['name']}")
            print(f"  Sample rate: {self.sample_rate} Hz")
            print(
                f"  Chunk size: {self.chunk_size} samples "
                f"({self.chunk_duration_ms}ms) [blocksize: {blocksize}]"
            )
            print(f"  Channels: {self.channels}")
            print(f"  Data type: {self.dtype}")

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
        dtype: str = "int16",
        normalize: bool = False,
    ):
        """
        Initialize file-based audio simulator.

        Args:
            audio_path: Path to audio file
            sample_rate: Target sample rate
            chunk_duration_ms: Duration of each chunk in milliseconds
            realtime: If True, simulate real-time by adding delays
            dtype: Data type for audio samples ('int16' or 'float32')
            normalize: If False, returns int16 audio like torchaudio.load(normalize=False)
        """
        import torchaudio

        self.audio_path = audio_path
        self.sample_rate = sample_rate
        self.chunk_duration_ms = chunk_duration_ms
        self.realtime = realtime
        self.dtype = dtype
        self.normalize = normalize

        # Load audio without normalization (returns int16)
        waveform, orig_sr = torchaudio.load(audio_path, normalize=False)

        # Resample if needed
        if orig_sr != sample_rate:
            resampler = torchaudio.transforms.Resample(orig_sr, sample_rate)
            waveform = resampler(waveform)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Convert to numpy
        self.audio = waveform.squeeze().numpy()

        # Convert dtype if needed
        if dtype == "int16" and self.audio.dtype != np.int16:
            self.audio = self.audio.astype(np.int16)
        elif dtype == "float32" and self.audio.dtype != np.float32:
            # If audio is int16, normalize to float32 [-1, 1]
            if self.audio.dtype == np.int16:
                self.audio = self.audio.astype(np.float32) / 32767.0
            else:
                self.audio = self.audio.astype(np.float32)

        self.chunk_size = int(sample_rate * chunk_duration_ms / 1000)
        self.position = 0
        self.is_running = False

        print(f"✓ Loaded audio file: {audio_path}")
        print(f"  Duration: {len(self.audio) / sample_rate:.2f}s")
        print(f"  Samples: {len(self.audio)}")
        print(f"  Data type: {self.audio.dtype}")

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

    # Prefer PyAudio on macOS for better stability
    capture_class: Type[Union[PyAudioStreamCapture, AudioStreamCapture]]
    if PYAUDIO_AVAILABLE:
        print("Using PyAudio backend")
        capture_class = PyAudioStreamCapture
    elif SOUNDDEVICE_AVAILABLE:
        print("Using sounddevice backend")
        capture_class = AudioStreamCapture
    else:
        raise ImportError("Neither PyAudio nor sounddevice is available")

    with capture_class(sample_rate=16000, chunk_duration_ms=480, dtype="int16") as capture:
        import time

        start = time.time()
        chunk_count = 0

        while time.time() - start < 5.0:
            chunk = capture.read_chunk()
            if chunk is not None:
                chunk_count += 1
                print(
                    f"Chunk {chunk_count}: {chunk.shape}, dtype: {chunk.dtype}, "
                    f"min: {chunk.min()}, max: {chunk.max()}, mean: {chunk.mean():.2f}"
                )

        print(f"\nCaptured {chunk_count} chunks in 5 seconds")
