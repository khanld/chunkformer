import os
import torch
import torchaudio
import yaml
import jiwer
import argparse
import pandas as pd

from tqdm import tqdm
from colorama import Fore, Style

import torchaudio.compliance.kaldi as kaldi
from model.utils.init_model import init_model
from model.utils.checkpoint import load_checkpoint
from model.utils.file_utils import read_symbol_table
from model.utils.ctc_utils import get_output_with_timestamps, get_output
from contextlib import nullcontext
from pydub import AudioSegment

@torch.no_grad()
def init(model_checkpoint, device):

    config_path = os.path.join(model_checkpoint, "config.yaml")
    checkpoint_path = os.path.join(model_checkpoint, "pytorch_model.bin")
    symbol_table_path = os.path.join(model_checkpoint, "vocab.txt")

    with open(config_path, 'r') as fin:
        config = yaml.load(fin, Loader=yaml.FullLoader)
    model = init_model(config, config_path)
    model.eval()
    load_checkpoint(model , checkpoint_path)

    model.encoder = model.encoder.to(device)
    model.ctc = model.ctc.to(device)
    # print('the number of encoder params: {:,d}'.format(num_params))

    symbol_table = read_symbol_table(symbol_table_path)
    char_dict = {v: k for k, v in symbol_table.items()}

    return model, char_dict

def load_audio(audio_path):
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_frame_rate(16000)
    audio = audio.set_sample_width(2)  # set bit depth to 16bit
    audio = audio.set_channels(1)  # set to mono
    audio = torch.as_tensor(audio.get_array_of_samples(), dtype=torch.float32).unsqueeze(0)
    return audio

@torch.no_grad()
def endless_decode(args, model, char_dict):    
    def get_max_input_context(c, r, n):
        return r + max(c, r) * (n-1)
    
    device = next(model.parameters()).device
    audio_path = args.long_form_audio
    # model configuration
    subsampling_factor = model.encoder.embed.subsampling_rate
    chunk_size = args.chunk_size
    left_context_size = args.left_context_size
    right_context_size = args.right_context_size
    conv_lorder = model.encoder.cnn_module_kernel // 2

    # get the maximum length that the gpu can consume
    max_length_limited_context = args.total_batch_duration
    max_length_limited_context = int((max_length_limited_context // 0.01))//2 # in 10ms second

    multiply_n = max_length_limited_context // chunk_size // subsampling_factor
    truncated_context_size = chunk_size * multiply_n # we only keep this part for text decoding

    # get the relative right context size
    rel_right_context_size = get_max_input_context(chunk_size, max(right_context_size, conv_lorder), model.encoder.num_blocks)
    rel_right_context_size = rel_right_context_size * subsampling_factor


    waveform = load_audio(audio_path)
    offset = torch.zeros(1, dtype=torch.int, device=device)

    # waveform = padding(waveform, sample_rate)
    xs = kaldi.fbank(waveform,
                            num_mel_bins=80,
                            frame_length=25,
                            frame_shift=10,
                            dither=0.0,
                            energy_floor=0.0,
                            sample_frequency=16000).unsqueeze(0)

    hyps = []
    att_cache = torch.zeros((model.encoder.num_blocks, left_context_size, model.encoder.attention_heads, model.encoder._output_size * 2 // model.encoder.attention_heads)).to(device)
    cnn_cache = torch.zeros((model.encoder.num_blocks, model.encoder._output_size, conv_lorder)).to(device)    # print(context_size)
    for idx, _ in tqdm(list(enumerate(range(0, xs.shape[1], truncated_context_size * subsampling_factor)))):
        start = max(truncated_context_size * subsampling_factor * idx, 0)
        end = min(truncated_context_size * subsampling_factor * (idx+1) + 7, xs.shape[1])

        x = xs[:, start:end+rel_right_context_size]
        x_len = torch.tensor([x[0].shape[0]], dtype=torch.int).to(device)

        encoder_outs, encoder_lens, _, att_cache, cnn_cache, offset = model.encoder.forward_parallel_chunk(xs=x, 
                                                                    xs_origin_lens=x_len, 
                                                                    chunk_size=chunk_size,
                                                                    left_context_size=left_context_size,
                                                                    right_context_size=right_context_size,
                                                                    att_cache=att_cache,
                                                                    cnn_cache=cnn_cache,
                                                                    truncated_context_size=truncated_context_size,
                                                                    offset=offset
                                                                    )
        encoder_outs = encoder_outs.reshape(1, -1, encoder_outs.shape[-1])[:, :encoder_lens]
        if chunk_size * multiply_n * subsampling_factor * idx + rel_right_context_size < xs.shape[1]:
            encoder_outs = encoder_outs[:, :truncated_context_size]  # (B, maxlen, vocab_size) # exclude the output of rel right context
        offset = offset - encoder_lens + encoder_outs.shape[1]


        hyp = model.encoder.ctc_forward(encoder_outs).squeeze(0)
        hyps.append(hyp)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if chunk_size * multiply_n * subsampling_factor * idx + rel_right_context_size >= xs.shape[1]:
            break
    hyps = torch.cat(hyps)
    decode = get_output_with_timestamps([hyps], char_dict)[0]

    for item in decode:
        start = f"{Fore.RED}{item['start']}{Style.RESET_ALL}"
        end = f"{Fore.RED}{item['end']}{Style.RESET_ALL}"
        print(f"{start} - {end}: {item['decode']}")


@torch.no_grad()
def batch_decode(args, model, char_dict):
    df = pd.read_csv(args.audio_list, sep="\t")

    max_length_limited_context = args.total_batch_duration
    max_length_limited_context = int((max_length_limited_context // 0.01)) // 2 # in 10ms second    xs = []
    max_frames = max_length_limited_context
    chunk_size = args.chunk_size
    left_context_size = args.left_context_size
    right_context_size = args.right_context_size
    device = next(model.parameters()).device

    decodes = []
    xs = []
    xs_origin_lens = []
    for idx, audio_path in tqdm(enumerate(df['wav'].to_list())):
        waveform = load_audio(audio_path)
        x = kaldi.fbank(waveform,
                                num_mel_bins=80,
                                frame_length=25,
                                frame_shift=10,
                                dither=0.0,
                                energy_floor=0.0,
                                sample_frequency=16000)

        xs.append(x)
        xs_origin_lens.append(x.shape[0])
        max_frames -= xs_origin_lens[-1]

        if (max_frames <= 0) or (idx == len(df) - 1):
            xs_origin_lens = torch.tensor(xs_origin_lens, dtype=torch.int, device=device)
            offset = torch.zeros(len(xs), dtype=torch.int, device=device)
            encoder_outs, encoder_lens, n_chunks, _, _, _ = model.encoder.forward_parallel_chunk(xs=xs, 
                                                                        xs_origin_lens=xs_origin_lens, 
                                                                        chunk_size=chunk_size,
                                                                        left_context_size=left_context_size,
                                                                        right_context_size=right_context_size,
                                                                        offset=offset
            )

            hyps = model.encoder.ctc_forward(encoder_outs, encoder_lens, n_chunks)
            decodes += get_output(hyps, char_dict)
                                         

            # reset
            xs = []
            xs_origin_lens = []
            max_frames = max_length_limited_context


    df['decode'] = decodes
    if "txt" in df:
        wer = jiwer.wer(df["txt"].to_list(), decodes)
        print("WER: ", wer)
    df.to_csv(args.audio_list, sep="\t", index=False)



def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description="Process arguments with default values.")

    # Add arguments with default values
    parser.add_argument(
        "--model_checkpoint", 
        type=str, 
        default=None, 
        help="Path to Huggingface checkpoint repo"
    )
    parser.add_argument(
        "--total_batch_duration", 
        type=int, 
        default=1800, 
        help="The total audio duration (in second) in a batch that your GPU memory can handle at once. Default is 1800s"
    )
    parser.add_argument(
        "--chunk_size", 
        type=int, 
        default=64, 
        help="Size of the chunks (default: 64)"
    )
    parser.add_argument(
        "--left_context_size", 
        type=int, 
        default=128, 
        help="Size of the left context (default: 128)"
    )
    parser.add_argument(
        "--right_context_size", 
        type=int, 
        default=128, 
        help="Size of the right context (default: 128)"
    )
    parser.add_argument(
        "--long_form_audio", 
        type=str, 
        default=None, 
        help="Path to the long audio file (default: None)"
    )
    parser.add_argument(
        "--audio_list", 
        type=str, 
        default=None, 
        required=False, 
        help="Path to the TSV file containing the audio list. The TSV file must have one column named 'wav'. If 'txt' column is provided, Word Error Rate (WER) is computed"
    )
    parser.add_argument(
        "--full_attn", 
        action="store_true",
        help="Whether to use full attention with caching. If not provided, limited-chunk attention will be used (default: False)"
    )
    parser.add_argument(
        "--device",
        type=torch.device,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the model on (default: cuda if available else cpu)"
    )
    parser.add_argument(
        "--autocast_dtype",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        default=None,
        help="Dtype for autocast. If not provided, autocast is disabled by default."
    )

    # Parse arguments
    args = parser.parse_args()
    device = torch.device(args.device)
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16, None: None}[args.autocast_dtype]

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
    assert args.long_form_audio or args.audio_list, "`long_form_audio` or `audio_list` must be activated"

    model, char_dict = init(args.model_checkpoint, device)
    with torch.autocast(device.type, dtype) if dtype is not None else nullcontext():
        if args.long_form_audio:
            endless_decode(args, model, char_dict)
        else:
            batch_decode(args, model, char_dict)

if __name__ == "__main__":
    main()

