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

from chunkformer.transducer.search.greedy_search import batch_greedy_search, optimized_search
from chunkformer.utils.checkpoint import load_checkpoint
from chunkformer.utils.file_utils import read_symbol_table
from chunkformer.utils.init_model import init_speech_model
from chunkformer.utils.model_utils import get_output, get_output_with_timestamps

# Import ChunkFormer modules

logger = logging.get_logger(__name__)


class ChunkFormerConfig(PretrainedConfig):
    """
    Configuration class for ChunkFormer model.
    """

    model_type = "chunkformer"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # set default decoder_conf and model_conf to {} if not exist
        if "decoder_conf" not in self.__dict__:
            self.decoder_conf = {}
        if "model_conf" not in self.__dict__:
            self.model_conf = {}
        if "model" not in self.__dict__:
            self.model = "asr_model"

    @classmethod
    def from_dict(cls, config_dict: dict, **kwargs):
        """Create config from dictionary."""
        config = cls(**kwargs)
        config.__dict__.update(config_dict)
        return config


class ChunkFormerModel(PreTrainedModel):
    """
    ChunkFormer model for Automatic Speech Recognition, compatible with Hugging Face transformers.
    """

    config_class = ChunkFormerConfig  # type: ignore[assignment]
    base_model_prefix = "chunkformer"
    main_input_name = "xs"
    supports_gradient_checkpointing = True

    def __init__(self, config):
        if isinstance(config, dict):
            # Convert dict to ChunkFormerConfig
            config = ChunkFormerConfig.from_dict(config)

        super().__init__(config)
        self.config = config

        # Initialize the model components directly (avoiding file path dependencies)
        self.model = self._init_model_from_config()
        self.char_dict = None  # Will be set when loading symbol table

        # Post-init
        self.post_init()

    def _init_model_from_config(self):
        """Initialize model from config."""
        # Convert config to dict for init_speech_model compatibility
        config_dict = self.config.__dict__.copy()

        # Handle CMVN configuration if file path exists
        if (
            hasattr(self.config, "cmvn_file")
            and self.config.cmvn_file
            and os.path.exists(self.config.cmvn_file)
        ):
            config_dict["cmvn"] = "global_cmvn"
            if "cmvn_conf" not in config_dict:
                config_dict["cmvn_conf"] = {}
            config_dict["cmvn_conf"]["cmvn_file"] = self.config.cmvn_file
            config_dict["cmvn_conf"]["is_json_cmvn"] = getattr(self.config, "is_json_cmvn", True)

        # Initialize model using init_speech_model with original YAML structure
        model, _ = init_speech_model(args=None, configs=config_dict)
        return model

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        config: Optional[dict] = None,
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
                with open(config_path, "r") as f:
                    config_dict = yaml.load(f, Loader=yaml.FullLoader)

                cmvn_path = os.path.join(pretrained_model_name_or_path, "global_cmvn")
                if os.path.exists(cmvn_path):
                    config_dict["cmvn_file"] = cmvn_path
                else:
                    logger.warning(
                        f"CMVN file {cmvn_path} not found in {pretrained_model_name_or_path}"
                    )
                    config_dict["cmvn_file"] = None

                # Convert dict to ChunkFormerConfig
                config = ChunkFormerConfig.from_dict(config_dict)
            else:
                raise ValueError(f"No config found in {pretrained_model_name_or_path}")
        elif isinstance(config, dict):
            # Convert dict to ChunkFormerConfig if needed
            config = ChunkFormerConfig.from_dict(config)

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
        **kwargs,
    ):
        """
        Forward method is not implemented for ChunkFormer.
        Use specific methods instead based on your use case.
        """
        raise NotImplementedError(
            "Forward method is not implemented. If you want to use ChunkFormer for feature "
            "extraction from pretrained model please use 'chunkformer.encode()' instead. "
            "Or if you want transcription please use 'endless_decode' or 'batch_decode'."
        )

    def get_encoder(self):
        """Get the encoder module."""
        return self.model.encoder

    def get_ctc(self):
        """Get the CTC module."""
        return self.model.ctc

    def encode(
        self,
        xs: torch.Tensor,
        xs_lens: torch.Tensor,
        chunk_size: Optional[int] = None,
        left_context_size: Optional[int] = None,
        right_context_size: Optional[int] = None,
        **kwargs,
    ):
        xs, xs_masks = self.model.encoder.forward_encoder(
            xs=xs,
            xs_lens=xs_lens,
            chunk_size=chunk_size,
            left_context_size=left_context_size,
            right_context_size=right_context_size,
            **kwargs,
        )
        xs_lens = xs_masks.squeeze(1).sum(-1)
        return xs, xs_lens

    def _load_audio_and_extract_features(self, audio_path: str):
        """
        Load audio file and extract fbank features using config parameters.

        Args:
            audio_path: Path to audio file

        Returns:
            torch.Tensor: Fbank features tensor
        """
        # Get config parameters with defaults
        fbank_conf = getattr(self.config, "fbank_conf", {})
        resample_conf = getattr(self.config, "resample_conf", {})

        # Extract parameters with fallback defaults
        sample_rate = resample_conf.get("resample_rate", 16000)
        num_mel_bins = fbank_conf.get("num_mel_bins", 80)
        frame_length = fbank_conf.get("frame_length", 25)
        frame_shift = fbank_conf.get("frame_shift", 10)
        dither = 0.0

        # Load audio
        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_frame_rate(sample_rate)
        audio = audio.set_sample_width(2)  # set bit depth to 16bit
        audio = audio.set_channels(1)  # set to mono
        waveform = torch.as_tensor(
            audio.get_array_of_samples(), dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        # Extract fbank features
        x = kaldi.fbank(
            waveform,
            num_mel_bins=num_mel_bins,
            frame_length=frame_length,
            frame_shift=frame_shift,
            dither=dither,
            energy_floor=0.0,
            sample_frequency=sample_rate,
        )
        x_len = x.shape[0]

        return x, x_len

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

        # Load audio and extract features using config parameters
        xs, xs_len = self._load_audio_and_extract_features(audio_path)
        xs = xs.unsqueeze(0)
        offset = torch.zeros(1, dtype=torch.int, device=device)

        encoder_outs = []
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
            list(enumerate(range(0, xs_len, truncated_context_size * subsampling_factor)))
        ):
            start = max(truncated_context_size * subsampling_factor * idx, 0)
            end = min(truncated_context_size * subsampling_factor * (idx + 1) + 7, xs_len)

            x = xs[:, start : end + rel_right_context_size]
            x_len = torch.tensor([x[0].shape[0]], dtype=torch.int).to(device)

            (
                encoder_out,
                encoder_len,
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

            encoder_out = encoder_out.reshape(1, -1, encoder_out.shape[-1])[:, :encoder_len]
            if chunk_size * multiply_n * subsampling_factor * idx + rel_right_context_size < xs_len:
                encoder_out = encoder_out[
                    :, :truncated_context_size
                ]  # exclude the output of rel right context
            offset = offset - encoder_len + encoder_out.shape[1]

            encoder_outs.append(encoder_out)

            if device.type == "cuda":
                torch.cuda.empty_cache()
            if (
                chunk_size * multiply_n * subsampling_factor * idx + rel_right_context_size
                >= xs_len
            ):
                break
        encoder_outs = torch.cat(encoder_outs, dim=1)  # [1, T, F]  # type: ignore[assignment]
        if self.config.model == "asr_model":
            token_predictions = self.model.ctc.log_softmax(encoder_outs).squeeze(0)  # [1, T, V]
            token_predictions = torch.argmax(token_predictions, dim=-1).reshape(1, -1, 1)
        else:
            encoder_len = torch.tensor(
                [encoder_outs.shape[1]], device=encoder_outs.device  # type: ignore[attr-defined]
            )
            token_predictions = optimized_search(
                self.model, encoder_out=encoder_outs, encoder_out_lens=encoder_len
            )
            token_predictions = token_predictions.reshape(
                1, encoder_outs.size(1), -1  # type: ignore[attr-defined]
            )

        if self.char_dict is not None:
            decode_result = get_output_with_timestamps(
                token_predictions, self.char_dict, self.config.model
            )[0]
            if not return_timestamps:
                decode_result = " ".join([item["decode"] for item in decode_result]).strip()
        else:
            decode_result = token_predictions

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
            # Load audio and extract features using config parameters
            x, x_len = self._load_audio_and_extract_features(audio_path)

            xs.append(x)
            xs_origin_lens.append(x_len)
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
                if self.config.model == "asr_model":
                    ctc_logits = self.model.ctc.log_softmax(encoder_outs)
                    hyps = torch.argmax(ctc_logits, dim=-1)
                    hyps = hyps.split(n_chunks, dim=0)
                    hyps = [hyp.flatten()[:x_len] for hyp, x_len in zip(hyps, encoder_lens)]
                    if self.char_dict is not None:
                        hyps = get_output(hyps, self.char_dict, self.config.model)
                else:
                    encoder_outs = encoder_outs.split(n_chunks, dim=0)
                    encoder_outs = [
                        enc_out.reshape(-1, enc_out.shape[-1])[:enc_len]
                        for enc_out, enc_len in zip(encoder_outs, encoder_lens)
                    ]
                    encoder_outs = torch.nn.utils.rnn.pad_sequence(encoder_outs, batch_first=True)
                    hyps = batch_greedy_search(
                        self.model, encoder_out=encoder_outs, encoder_out_lens=encoder_lens
                    )
                    if self.char_dict is not None:
                        hyps = get_output(hyps, self.char_dict, self.config.model)

                decodes.extend(hyps)

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
                print(f"Word Error Rate (WER): {wer:.4f}")  # noqa: E231

            # Save results
            df.to_csv(args.audio_list, sep="\t", index=False)
            print(f"Results saved to {args.audio_list}")


if __name__ == "__main__":
    main()
