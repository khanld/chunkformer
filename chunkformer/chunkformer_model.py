"""
Hugging Face compatible ChunkFormer implementation
"""

import argparse
import os
from contextlib import nullcontext
from typing import List, Optional, Union

import jiwer
import pandas as pd
import torch
import torchaudio.compliance.kaldi as kaldi
import yaml
from colorama import Fore, Style
from huggingface_hub import snapshot_download
from pydub import AudioSegment
from tqdm import tqdm
from transformers import PretrainedConfig, PreTrainedModel
from transformers.utils import logging

from .model.utils.checkpoint import load_checkpoint
from .model.utils.ctc_utils import get_output, get_output_with_timestamps
from .model.utils.file_utils import read_symbol_table

# Import ChunkFormer modules

logger = logging.get_logger(__name__)


class ChunkFormerConfig(PretrainedConfig):
    """
    Configuration class for ChunkFormer model.
    """

    model_type = "chunkformer"

    def __init__(
        self,
        vocab_size: int = 4992,
        input_dim: int = 80,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 12,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
        activation_type: str = "swish",
        use_cnn_module: bool = True,
        cnn_module_kernel: int = 15,
        chunk_size: int = 64,
        left_context_size: int = 128,
        right_context_size: int = 128,
        dynamic_chunk_sizes: Optional[list] = None,
        dynamic_left_context_sizes: Optional[list] = None,
        dynamic_right_context_sizes: Optional[list] = None,
        ctc_weight: float = 0.3,
        cmvn_file: Optional[str] = None,
        is_json_cmvn: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.input_dim = input_dim
        self.output_size = output_size
        self.attention_heads = attention_heads
        self.linear_units = linear_units
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate
        self.positional_dropout_rate = positional_dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.activation_type = activation_type
        self.use_cnn_module = use_cnn_module
        self.cnn_module_kernel = cnn_module_kernel
        self.chunk_size = chunk_size
        self.left_context_size = left_context_size
        self.right_context_size = right_context_size
        self.dynamic_chunk_sizes = dynamic_chunk_sizes or [-1, -1, 64, 128, 256]
        self.dynamic_left_context_sizes = dynamic_left_context_sizes or [64, 128, 256]
        self.dynamic_right_context_sizes = dynamic_right_context_sizes or [64, 128, 256]
        self.ctc_weight = ctc_weight
        self.cmvn_file = cmvn_file
        self.is_json_cmvn = is_json_cmvn

    @classmethod
    def from_yaml_config(cls, yaml_path: str, **kwargs):
        """Create config from original ChunkFormer YAML config file."""
        with open(yaml_path, "r") as f:
            yaml_config = yaml.load(f, Loader=yaml.FullLoader)

        # Map YAML config to our config
        config_dict = {
            "vocab_size": yaml_config.get("output_dim", 4992),
            "input_dim": yaml_config.get("input_dim", 80),
            "cmvn_file": yaml_config.get("cmvn_file"),
            "is_json_cmvn": yaml_config.get("is_json_cmvn", True),
        }

        # Extract encoder config
        encoder_conf = yaml_config.get("encoder_conf", {})
        config_dict.update(
            {
                "output_size": encoder_conf.get("output_size", 256),
                "attention_heads": encoder_conf.get("attention_heads", 4),
                "linear_units": encoder_conf.get("linear_units", 2048),
                "num_blocks": encoder_conf.get("num_blocks", 12),
                "dropout_rate": encoder_conf.get("dropout_rate", 0.1),
                "positional_dropout_rate": encoder_conf.get("positional_dropout_rate", 0.1),
                "attention_dropout_rate": encoder_conf.get("attention_dropout_rate", 0.1),
                "activation_type": encoder_conf.get("activation_type", "swish"),
                "use_cnn_module": encoder_conf.get("use_cnn_module", True),
                "cnn_module_kernel": encoder_conf.get("cnn_module_kernel", 15),
                "dynamic_chunk_sizes": encoder_conf.get("dynamic_chunk_sizes"),
                "dynamic_left_context_sizes": encoder_conf.get("dynamic_left_context_sizes"),
                "dynamic_right_context_sizes": encoder_conf.get("dynamic_right_context_sizes"),
            }
        )

        # Extract model config
        model_conf = yaml_config.get("model_conf", {})
        config_dict.update(
            {
                "ctc_weight": model_conf.get("ctc_weight", 0.3),
            }
        )

        config_dict.update(kwargs)
        return cls(**config_dict)


class ChunkFormerModel(PreTrainedModel):
    """
    ChunkFormer model for Automatic Speech Recognition, compatible with Hugging Face transformers.
    """

    config_class = ChunkFormerConfig  # type: ignore[assignment]
    base_model_prefix = "chunkformer"
    main_input_name = "features"
    supports_gradient_checkpointing = True

    def __init__(self, config: ChunkFormerConfig):
        super().__init__(config)
        self.config = config

        # Initialize the model components directly (avoiding file path dependencies)
        self.model = self._init_model_from_config()

        # Store vocabulary
        self.vocab_size = config.vocab_size
        self.char_dict = None  # Will be set when loading symbol table

        # Post-init
        self.post_init()

    def _init_model_from_config(self):
        """Initialize model from config without file dependencies."""
        from .model.asr_model import ASRModel
        from .model.cmvn import GlobalCMVN
        from .model.ctc import CTC
        from .model.encoder import ChunkFormerEncoder
        from .model.utils.cmvn import load_cmvn

        # Handle CMVN
        global_cmvn = None
        if self.config.cmvn_file and os.path.exists(self.config.cmvn_file):
            try:
                mean, istd = load_cmvn(self.config.cmvn_file, self.config.is_json_cmvn)
                global_cmvn = GlobalCMVN(
                    torch.from_numpy(mean).float(), torch.from_numpy(istd).float()
                )
                print(f"Loaded CMVN from {self.config.cmvn_file}")
            except Exception as e:
                print(f"Warning: Failed to load CMVN from {self.config.cmvn_file}: {e}")
                global_cmvn = None

        input_dim = self.config.input_dim
        vocab_size = self.config.vocab_size

        # Get encoder config
        encoder_conf = {
            "output_size": self.config.output_size,
            "attention_heads": self.config.attention_heads,
            "linear_units": self.config.linear_units,
            "num_blocks": self.config.num_blocks,
            "dropout_rate": self.config.dropout_rate,
            "positional_dropout_rate": self.config.positional_dropout_rate,
            "attention_dropout_rate": self.config.attention_dropout_rate,
            "input_layer": "dw_striding",
            "pos_enc_layer_type": "chunk_rel_pos",
            "normalize_before": True,
            "selfattention_layer_type": "chunk_rel_seflattn",
            "activation_type": self.config.activation_type,
            "use_cnn_module": self.config.use_cnn_module,
            "cnn_module_kernel": self.config.cnn_module_kernel,
            "cnn_module_norm": "layer_norm",
            "dynamic_conv": True,
            "layer_norm_type": "layer_norm",
            "gradient_checkpointing": False,
            "final_norm": True,
            "norm_eps": 1e-5,
            "use_sdpa": False,
            "dynamic_chunk_sizes": self.config.dynamic_chunk_sizes,
            "dynamic_left_context_sizes": self.config.dynamic_left_context_sizes,
            "dynamic_right_context_sizes": self.config.dynamic_right_context_sizes,
        }

        # Initialize encoder
        encoder = ChunkFormerEncoder(input_dim, global_cmvn=global_cmvn, **encoder_conf)

        # Initialize CTC
        ctc = CTC(vocab_size, encoder._output_size)

        # Initialize full model
        model = ASRModel(vocab_size=vocab_size, encoder=encoder, ctc=ctc)

        return model

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args,
        config: Optional[ChunkFormerConfig] = None,
        cache_dir: Optional[str] = None,
        force_download: bool = False,
        **kwargs,
    ):
        """
        Load a pretrained ChunkFormer model.

        Args:
            pretrained_model_name_or_path: Path to the local pretrained model or HuggingFace model
            config: Model configuration
            cache_dir: Directory to cache downloaded models
            force_download: Whether to force download even if cached
            **kwargs: Additional arguments
        """
        # Check if it's a local path or HuggingFace model identifier
        is_local = os.path.isdir(pretrained_model_name_or_path)

        if not is_local:
            # Try to download from HuggingFace Hub
            try:
                logger.info(
                    f"Downloading model from HuggingFace Hub: {pretrained_model_name_or_path}"
                )
                model_path = snapshot_download(
                    repo_id=pretrained_model_name_or_path,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    **kwargs,
                )
                pretrained_model_name_or_path = model_path
                logger.info(f"Model downloaded to: {model_path}")
            except Exception as e:
                logger.warning(f"Failed to download from HuggingFace Hub: {e}")

        # If config is not provided, try to load from config files
        if config is None:
            # Try config.yaml first (original ChunkFormer format)
            config_path = os.path.join(pretrained_model_name_or_path, "config.yaml")
            if os.path.exists(config_path):
                config = ChunkFormerConfig.from_yaml_config(config_path)
                cmvn_path = os.path.join(pretrained_model_name_or_path, "global_cmvn")
                if os.path.exists(cmvn_path):
                    config.cmvn_file = cmvn_path
                else:
                    logger.warning(
                        f"CMVN file {config.cmvn_file} not found in {pretrained_model_name_or_path}"
                    )
                    config.cmvn_file = None
            else:
                # Try to load from config.json (HuggingFace format)
                config_json_path = os.path.join(pretrained_model_name_or_path, "config.json")
                if os.path.exists(config_json_path):
                    config = ChunkFormerConfig.from_json_file(config_json_path)
                else:
                    raise ValueError(f"No config found in {pretrained_model_name_or_path}")

        # Initialize model
        model = cls(config)

        # Load weights - try multiple checkpoint formats
        checkpoint_candidates = ["pytorch_model.bin", "pytorch_model.pt", "pytorch_model.ckpt"]

        checkpoint_path = None
        for candidate in checkpoint_candidates:
            candidate_path = os.path.join(pretrained_model_name_or_path, candidate)
            if os.path.exists(candidate_path):
                checkpoint_path = candidate_path
                break

        if checkpoint_path is None:
            raise ValueError(
                f"No checkpoint found in {pretrained_model_name_or_path}. "
                f"Expected one of: {checkpoint_candidates}"
            )

        # Load checkpoint using original ChunkFormer loading function
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        load_checkpoint(model.model, checkpoint_path)
        model.eval()  # Set the entire model to eval mode

        # Load symbol table if available
        vocab_path = os.path.join(pretrained_model_name_or_path, "vocab.txt")
        if os.path.exists(vocab_path):
            symbol_table = read_symbol_table(vocab_path)
            model.char_dict = {v: k for k, v in symbol_table.items()}  # type: ignore[assignment]

        return model

    def save_pretrained(self, save_directory: Union[str, os.PathLike], **kwargs):  # type: ignore
        """Save the model to a directory."""
        os.makedirs(save_directory, exist_ok=True)

        # Save config
        self.config.save_pretrained(save_directory)

        # Save model weights
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.model.state_dict(), model_path)

        logger.info(f"Model saved to {save_directory}")

    def forward(
        self,
        features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
        left_context_size: Optional[int] = None,
        right_context_size: Optional[int] = None,
        **kwargs,
    ):
        """
        Forward pass of the model.

        Args:
            features: Input features of shape (batch_size, seq_len, feature_dim) or list of tensors
            attention_mask: Attention mask
            chunk_size: Chunk size for chunked attention
            left_context_size: Left context size
            right_context_size: Right context size
        """
        # Handle both tensor and list inputs
        if isinstance(features, torch.Tensor):
            batch_size, seq_len, _ = features.shape
            xs = [features[i] for i in range(batch_size)]

            # Create lengths tensor if not provided via attention_mask
            if attention_mask is not None:
                feature_lengths = attention_mask.sum(dim=1)
            else:
                feature_lengths = torch.full(
                    (batch_size,), seq_len, dtype=torch.long, device=features.device
                )
        else:
            # features is a list of tensors
            xs = features
            feature_lengths = torch.tensor(
                [x.shape[0] for x in xs], dtype=torch.long, device=xs[0].device
            )

        # Use default chunk sizes if not provided
        chunk_size = chunk_size or self.config.chunk_size
        left_context_size = left_context_size or self.config.left_context_size
        right_context_size = right_context_size or self.config.right_context_size

        # Initialize cache and offset
        device = xs[0].device
        offset = torch.zeros(len(xs), dtype=torch.int, device=device)

        # Forward through the encoder using forward_parallel_chunk
        encoder_out, encoder_lens, n_chunks, _, _, _ = self.model.encoder.forward_parallel_chunk(
            xs=xs,
            xs_origin_lens=feature_lengths,
            chunk_size=chunk_size,
            left_context_size=left_context_size,
            right_context_size=right_context_size,
            offset=offset,
        )

        # Get CTC outputs using our wrapper method
        ctc_logits = self.ctc_forward(encoder_out, encoder_lens, n_chunks)

        return {
            "logits": ctc_logits,
            "encoder_outputs": encoder_out,
            "encoder_lengths": encoder_lens,
            "n_chunks": n_chunks,
        }

    def get_encoder(self):
        """Get the encoder module."""
        return self.model.encoder

    def get_ctc(self):
        """Get the CTC module."""
        return self.model.ctc

    def encode(
        self,
        features: Union[torch.Tensor, list],
        feature_lengths: torch.Tensor,
        chunk_size: Optional[int] = None,
        left_context_size: Optional[int] = None,
        right_context_size: Optional[int] = None,
    ):
        """Encode features using the encoder with forward_parallel_chunk."""
        chunk_size = chunk_size or self.config.chunk_size
        left_context_size = left_context_size or self.config.left_context_size
        right_context_size = right_context_size or self.config.right_context_size

        # Convert tensor to list if needed
        if isinstance(features, torch.Tensor):
            xs = [features[i] for i in range(features.shape[0])]
        else:
            xs = features

        device = xs[0].device
        offset = torch.zeros(len(xs), dtype=torch.int, device=device)

        return self.model.encoder.forward_parallel_chunk(
            xs=xs,
            xs_origin_lens=feature_lengths,
            chunk_size=chunk_size,
            left_context_size=left_context_size,
            right_context_size=right_context_size,
            offset=offset,
        )

    def ctc_forward(self, encoder_out, encoder_lens=None, n_chunks=None):
        """Apply CTC to encoder outputs."""
        # Always return raw logits for HF interface
        return self.model.ctc.log_softmax(encoder_out)

    @torch.no_grad()
    def endless_decode(
        self,
        audio_path: str,
        chunk_size: Optional[int] = 64,
        left_context_size: Optional[int] = 128,
        right_context_size: Optional[int] = 128,
        total_batch_duration: int = 1800,
        return_timestamps: bool = True,
    ):
        """
        Perform streaming/endless decoding on long-form audio.

        Args:
            audio_path: Path to audio file
            chunk_size: Chunk size for processing
            left_context_size: Left context size
            right_context_size: Right context size
            total_batch_duration: Total duration in seconds for batch processing
            return_timestamps: Whether to return timestamps
        """

        def get_max_input_context(c, r, n):
            return r + max(c, r) * (n - 1)

        def load_audio(audio_path):
            audio = AudioSegment.from_file(audio_path)
            audio = audio.set_frame_rate(16000)
            audio = audio.set_sample_width(2)  # set bit depth to 16bit
            audio = audio.set_channels(1)  # set to mono
            audio = torch.as_tensor(audio.get_array_of_samples(), dtype=torch.float32).unsqueeze(0)
            return audio

        device = next(self.parameters()).device

        # Use config defaults if not provided
        chunk_size = chunk_size if chunk_size is not None else 64
        left_context_size = left_context_size if left_context_size is not None else 128
        right_context_size = right_context_size if right_context_size is not None else 128

        # Model configuration
        subsampling_factor = self.model.encoder.embed.subsampling_rate
        conv_lorder = self.model.encoder.cnn_module_kernel // 2

        # Get the maximum length that the gpu can consume
        max_length_limited_context = total_batch_duration
        max_length_limited_context = (
            int((max_length_limited_context // 0.01)) // 2
        )  # in 10ms second

        multiply_n = max_length_limited_context // chunk_size // subsampling_factor
        truncated_context_size = chunk_size * multiply_n  # we only keep this part for text decoding

        # Get the relative right context size
        rel_right_context_size = get_max_input_context(
            chunk_size, max(right_context_size, conv_lorder), self.model.encoder.num_blocks
        )
        rel_right_context_size = rel_right_context_size * subsampling_factor

        # Load and preprocess audio
        waveform = load_audio(audio_path)
        offset = torch.zeros(1, dtype=torch.int, device=device)

        # Extract features
        xs = kaldi.fbank(
            waveform,
            num_mel_bins=80,
            frame_length=25,
            frame_shift=10,
            dither=0.0,
            energy_floor=0.0,
            sample_frequency=16000,
        ).unsqueeze(0)

        hyps = []
        att_cache = torch.zeros(
            (
                self.model.encoder.num_blocks,
                left_context_size,
                self.model.encoder.attention_heads,
                self.model.encoder._output_size * 2 // self.model.encoder.attention_heads,
            )
        ).to(device)
        cnn_cache = torch.zeros(
            (self.model.encoder.num_blocks, self.model.encoder._output_size, conv_lorder)
        ).to(device)

        for idx, _ in tqdm(
            list(enumerate(range(0, xs.shape[1], truncated_context_size * subsampling_factor)))
        ):
            start = max(truncated_context_size * subsampling_factor * idx, 0)
            end = min(truncated_context_size * subsampling_factor * (idx + 1) + 7, xs.shape[1])

            x = xs[:, start : end + rel_right_context_size]
            x_len = torch.tensor([x[0].shape[0]], dtype=torch.int).to(device)

            (
                encoder_outs,
                encoder_lens,
                _,
                att_cache,
                cnn_cache,
                offset,
            ) = self.model.encoder.forward_parallel_chunk(
                xs=[x.squeeze(0)],
                xs_origin_lens=x_len,
                chunk_size=chunk_size,
                left_context_size=left_context_size,
                right_context_size=right_context_size,
                att_cache=att_cache,
                cnn_cache=cnn_cache,
                truncated_context_size=truncated_context_size,
                offset=offset,
            )

            encoder_outs = encoder_outs.reshape(1, -1, encoder_outs.shape[-1])[:, :encoder_lens]
            if (
                chunk_size * multiply_n * subsampling_factor * idx + rel_right_context_size
                < xs.shape[1]
            ):
                encoder_outs = encoder_outs[
                    :, :truncated_context_size
                ]  # exclude the output of rel right context
            offset = offset - encoder_lens + encoder_outs.shape[1]

            hyp = self.model.ctc.log_softmax(encoder_outs).squeeze(0)
            hyps.append(hyp)

            if device.type == "cuda":
                torch.cuda.empty_cache()
            if (
                chunk_size * multiply_n * subsampling_factor * idx + rel_right_context_size
                >= xs.shape[1]
            ):
                break

        hyps = torch.cat(hyps)  # type: ignore[assignment]

        if return_timestamps and self.char_dict is not None:
            # Convert logits to token predictions
            token_predictions = torch.argmax(hyps, dim=-1)
            decode_result = get_output_with_timestamps([token_predictions], self.char_dict)[0]
        elif self.char_dict is not None:
            # Convert logits to token predictions
            token_predictions = torch.argmax(hyps, dim=-1)
            decode_result = get_output([token_predictions], self.char_dict)[0]
        else:
            decode_result = hyps

        return decode_result

    @torch.no_grad()
    def batch_decode(
        self,
        audio_paths: List[str],
        chunk_size: Optional[int] = 64,
        left_context_size: Optional[int] = 128,
        right_context_size: Optional[int] = 128,
        total_batch_duration: int = 1800,
    ):
        """
        Perform batch decoding on multiple audio samples.

        Args:
            audio_paths: List of paths to audio files
            chunk_size: Chunk size for processing
            left_context_size: Left context size
            right_context_size: Right context size
            total_batch_duration: Total duration in seconds for batch processing
        """

        def load_audio(audio_path):
            audio = AudioSegment.from_file(audio_path)
            audio = audio.set_frame_rate(16000)
            audio = audio.set_sample_width(2)  # set bit depth to 16bit
            audio = audio.set_channels(1)  # set to mono
            audio = torch.as_tensor(audio.get_array_of_samples(), dtype=torch.float32).unsqueeze(0)
            return audio

        max_length_limited_context = total_batch_duration
        max_length_limited_context = (
            int((max_length_limited_context // 0.01)) // 2
        )  # in 10ms second
        max_frames = max_length_limited_context

        chunk_size = chunk_size if chunk_size is not None else 64
        left_context_size = left_context_size if left_context_size is not None else 128
        right_context_size = right_context_size if right_context_size is not None else 128
        device = next(self.parameters()).device

        decodes = []
        xs = []
        xs_origin_lens = []

        for idx, audio_path in tqdm(enumerate(audio_paths)):
            waveform = load_audio(audio_path)
            x = kaldi.fbank(
                waveform,
                num_mel_bins=80,
                frame_length=25,
                frame_shift=10,
                dither=0.0,
                energy_floor=0.0,
                sample_frequency=16000,
            )

            xs.append(x)
            xs_origin_lens.append(x.shape[0])
            max_frames -= xs_origin_lens[-1]

            if (max_frames <= 0) or (idx == len(audio_paths) - 1):
                xs_origin_lens = torch.tensor(
                    xs_origin_lens, dtype=torch.int, device=device
                )  # type: ignore[assignment]
                offset = torch.zeros(len(xs), dtype=torch.int, device=device)

                (
                    encoder_outs,
                    encoder_lens,
                    n_chunks,
                    _,
                    _,
                    _,
                ) = self.model.encoder.forward_parallel_chunk(
                    xs=xs,
                    xs_origin_lens=xs_origin_lens,
                    chunk_size=chunk_size,
                    left_context_size=left_context_size,
                    right_context_size=right_context_size,
                    offset=offset,
                )

                # Get CTC logits
                ctc_logits = self.model.ctc.log_softmax(encoder_outs)
                # Convert to token predictions for decoding
                hyps = torch.argmax(ctc_logits, dim=-1)

                if self.char_dict is not None:
                    # Split by chunks if needed
                    if n_chunks is not None and encoder_lens is not None:
                        hyps_split = hyps.split(n_chunks, dim=0)
                        hyps_list = [
                            hyp.flatten()[:x_len] for hyp, x_len in zip(hyps_split, encoder_lens)
                        ]
                    else:
                        hyps_list = [hyp for hyp in hyps]

                    batch_decodes = get_output(hyps_list, self.char_dict)
                    decodes.extend(batch_decodes)
                else:
                    decodes.extend([hyp for hyp in hyps])

                # Reset
                xs = []
                xs_origin_lens = []
                max_frames = max_length_limited_context

        return decodes


# Register the configuration and model
ChunkFormerConfig.register_for_auto_class()
ChunkFormerModel.register_for_auto_class("AutoModel")


def main():
    """Main function for command line interface."""
    # Create argument parser
    parser = argparse.ArgumentParser(
        description="ChunkFormer ASR inference with command line interface."
    )

    # Add arguments with default values
    parser.add_argument(
        "--model_checkpoint", type=str, default=None, help="Path to Huggingface checkpoint repo"
    )
    parser.add_argument(
        "--total_batch_duration",
        type=int,
        default=1800,
        help="The total audio duration (in second) in a batch \
        that your GPU memory can handle at once. Default is 1800s",
    )
    parser.add_argument(
        "--chunk_size", type=int, default=64, help="Size of the chunks (default: 64)"
    )
    parser.add_argument(
        "--left_context_size", type=int, default=128, help="Size of the left context (default: 128)"
    )
    parser.add_argument(
        "--right_context_size",
        type=int,
        default=128,
        help="Size of the right context (default: 128)",
    )
    parser.add_argument(
        "--long_form_audio",
        type=str,
        default=None,
        help="Path to the long audio file (default: None)",
    )
    parser.add_argument(
        "--audio_list",
        type=str,
        default=None,
        required=False,
        help="Path to the TSV file containing the audio list. \
            The TSV file must have one column named 'wav'. \
            If 'txt' column is provided, Word Error Rate (WER) is computed",
    )
    parser.add_argument(
        "--full_attn",
        action="store_true",
        help="Whether to use full attention with caching. \
        If not provided, limited-chunk attention will be used (default: False)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the model on (default: cuda if available else cpu)",
    )
    parser.add_argument(
        "--autocast_dtype",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        default=None,
        help="Dtype for autocast. If not provided, autocast is disabled by default.",
    )

    # Parse arguments
    args = parser.parse_args()
    device = torch.device(args.device)
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16, None: None}[
        args.autocast_dtype
    ]

    # Print the arguments
    print(f"Model Checkpoint: {args.model_checkpoint}")
    print(f"Device: {device}")
    print(f"Total Duration in a Batch (in second): {args.total_batch_duration}")
    print(f"Chunk Size: {args.chunk_size}")
    print(f"Left Context Size: {args.left_context_size}")
    print(f"Right Context Size: {args.right_context_size}")
    print(f"Long Form Audio Path: {args.long_form_audio}")
    print(f"Audio List Path: {args.audio_list}")

    assert args.model_checkpoint is not None, "You must specify the path to the model"
    assert (
        args.long_form_audio or args.audio_list
    ), "`long_form_audio` or `audio_list` must be activated"

    # Load model using HuggingFace interface
    print("Loading model using HuggingFace interface...")
    model = ChunkFormerModel.from_pretrained(args.model_checkpoint)
    model = model.to(device)
    model.eval()

    # Perform inference
    with torch.autocast(device.type, dtype) if dtype is not None else nullcontext():
        if args.long_form_audio:
            decode = model.endless_decode(
                args.long_form_audio,
                chunk_size=args.chunk_size,
                left_context_size=args.left_context_size,
                right_context_size=args.right_context_size,
                total_batch_duration=args.total_batch_duration,
            )
            for item in decode:
                start = f"{Fore.RED}{item['start']}{Style.RESET_ALL}"
                end = f"{Fore.RED}{item['end']}{Style.RESET_ALL}"
                print(f"{start} - {end}: {item['decode']}")
        else:
            # Batch decode using HF model interface
            df = pd.read_csv(args.audio_list, sep="\t")
            audio_paths = df["wav"].to_list()

            decodes = model.batch_decode(
                audio_paths,
                chunk_size=args.chunk_size,
                left_context_size=args.left_context_size,
                right_context_size=args.right_context_size,
                total_batch_duration=args.total_batch_duration,
            )
            df["decode"] = decodes
            if "txt" in df.columns:
                wer = jiwer.wer(df["txt"].to_list(), decodes)
                print(f"Word Error Rate (WER): {wer:.4f}")

            # Save results
            df.to_csv(args.audio_list, sep="\t", index=False)
            print(f"Results saved to {args.audio_list}")


if __name__ == "__main__":
    main()
