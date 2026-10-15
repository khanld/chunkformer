"""Regression tests for native TorchCodec audio decoding and Kaldi feature scaling."""

import math
import wave
from pathlib import Path

import pytest
import torch
import torchaudio.compliance.kaldi as kaldi
from torchcodec.decoders import AudioDecoder

from chunkformer.dataset.processor import decode_wav


@pytest.fixture
def wav_path(tmp_path: Path) -> Path:
    path = tmp_path / "stereo.wav"
    frames = bytearray()
    for i in range(8000):
        left = int(12000 * math.sin(2 * math.pi * 440 * i / 8000))
        right = int(6000 * math.sin(2 * math.pi * 440 * i / 8000))
        frames.extend(left.to_bytes(2, "little", signed=True))
        frames.extend(right.to_bytes(2, "little", signed=True))
    with wave.open(str(path), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(8000)
        output.writeframes(frames)
    return path


def test_decode_wav_path_and_bytes(wav_path: Path):
    for source in (str(wav_path), wav_path.read_bytes()):
        sample = decode_wav({"key": "clip", "wav": source})
        assert sample["sample_rate"] == 8000
        assert sample["wav"].shape == (2, 8000)
        assert sample["wav"].dtype == torch.float32
        assert sample["wav"].abs().max() <= 1.0


def test_decode_wav_range(wav_path: Path):
    sample = decode_wav({"key": "clip", "wav": wav_path.read_bytes(), "start": 0.25, "end": 0.75})
    assert sample["sample_rate"] == 8000
    assert sample["wav"].shape[0] == 2
    assert abs(sample["wav"].shape[1] / sample["sample_rate"] - 0.5) < 0.01


def test_model_feature_extraction_uses_pcm_scale(wav_path: Path):
    from types import SimpleNamespace

    from chunkformer.chunkformer_model import ChunkFormerModel

    # Exercise the real audio entry point without downloading model weights.
    model = SimpleNamespace(
        config=SimpleNamespace(
            fbank_conf={"num_mel_bins": 80},
            resample_conf={"resample_rate": 16000},
        ),
        device=torch.device("cpu"),
    )
    features, length = ChunkFormerModel._load_audio_and_extract_features(model, str(wav_path))
    assert features.shape == (length, 80)
    assert length > 0
    assert torch.isfinite(features).all()

    decoded = AudioDecoder(str(wav_path), sample_rate=16000, num_channels=1).get_all_samples()
    expected = kaldi.fbank(
        decoded.data * (1 << 15),
        sample_frequency=16000,
        num_mel_bins=80,
        frame_length=25,
        frame_shift=10,
        dither=0.0,
        energy_floor=0.0,
    )
    torch.testing.assert_close(features.cpu(), expected, rtol=1e-4, atol=1e-4)


def test_speed_perturb_changes_duration_without_changing_sample_rate(wav_path: Path):
    from chunkformer.dataset.processor import speed_perturb

    source = decode_wav({"key": "clip", "wav": str(wav_path)})
    original_duration = source["wav"].shape[-1] / source["sample_rate"]
    result = speed_perturb(source, speeds=[1.1])
    new_duration = result["wav"].shape[-1] / result["sample_rate"]

    assert result["sample_rate"] == 8000
    assert new_duration == pytest.approx(original_duration / 1.1, rel=0.02)
    assert torch.isfinite(result["wav"]).all()


def test_distortion_audio_file_round_trip(wav_path: Path, tmp_path: Path):
    from chunkformer.dataset.wav_distortion import distort_wav_conf_and_save

    output = tmp_path / "distorted.wav"
    distort_wav_conf_and_save("none_distortion", None, 0.0, str(wav_path), str(output))
    decoded = AudioDecoder(str(output)).get_all_samples()

    assert output.is_file()
    assert decoded.sample_rate == 8000
    assert decoded.data.shape[0] == 2
    assert abs(decoded.data.shape[1] / decoded.sample_rate - 1.0) < 0.01
