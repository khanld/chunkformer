# Copyright (c) 2021 Mobvoi Inc (Binbin Zhang, Di Wu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import List, Tuple

import numpy as np
import torch
import torchaudio.functional as F


def remove_duplicates_and_blank(hyp: List[int], blank_id: int = 0) -> List[int]:
    new_hyp: List[int] = []
    cur = 0
    while cur < len(hyp):
        if hyp[cur] != blank_id:
            new_hyp.append(hyp[cur])
        prev = cur
        while cur < len(hyp) and hyp[cur] == hyp[prev]:
            cur += 1
    return new_hyp


def replace_duplicates_with_blank(hyp: List[int], blank_id: int = 0) -> List[int]:
    new_hyp: List[int] = []
    cur = 0
    while cur < len(hyp):
        new_hyp.append(hyp[cur])
        prev = cur
        cur += 1
        while cur < len(hyp) and hyp[cur] == hyp[prev] and hyp[cur] != blank_id:
            new_hyp.append(blank_id)
            cur += 1
    return new_hyp


def gen_ctc_peak_time(hyp: List[int], blank_id: int = 0) -> List[int]:
    times = []
    cur = 0
    while cur < len(hyp):
        if hyp[cur] != blank_id:
            times.append(cur)
        prev = cur
        while cur < len(hyp) and hyp[cur] == hyp[prev]:
            cur += 1
    return times


def gen_timestamps_from_peak(
    peaks: List[int],
    max_duration: float,
    frame_rate: float = 0.04,
    max_token_duration: float = 1.0,
) -> List[Tuple[float, float]]:
    """
    Args:
        peaks: ctc peaks time stamp
        max_duration: max_duration of the sentence
        frame_rate: frame rate of every time stamp, in seconds
        max_token_duration: max duration of the token, in seconds
    Returns:
        list(start, end) of each token
    """
    times = []
    half_max = max_token_duration / 2
    for i in range(len(peaks)):
        if i == 0:
            start = max(0, peaks[0] * frame_rate - half_max)
        else:
            start = max(
                (peaks[i - 1] + peaks[i]) / 2 * frame_rate, peaks[i] * frame_rate - half_max
            )

        if i == len(peaks) - 1:
            end = min(max_duration, peaks[-1] * frame_rate + half_max)
        else:
            end = min((peaks[i] + peaks[i + 1]) / 2 * frame_rate, peaks[i] * frame_rate + half_max)
        times.append((start, end))
    return times


def insert_blank(label, blank_id=0):
    """Insert blank token between every two label token."""
    label = np.expand_dims(label, 1)
    blanks = np.zeros((label.shape[0], 1), dtype=np.int64) + blank_id
    label = np.concatenate([blanks, label], axis=1)
    label = label.reshape(-1)
    label = np.append(label, label[0])
    return label


def force_align(ctc_probs: torch.Tensor, y: torch.Tensor, blank_id=0) -> list[int]:
    """ctc forced alignment.

    Args:
        torch.Tensor ctc_probs: hidden state sequence, 2d tensor (T, D)
        torch.Tensor y: id sequence tensor 1d tensor (L)
        int blank_id: blank symbol index
    Returns:
        torch.Tensor: alignment result
    """
    ctc_probs = ctc_probs[None].cpu()
    y = y[None].cpu()
    alignments, _ = F.forced_align(ctc_probs, y, blank=blank_id)
    result: list[int] = alignments[0].tolist()
    return result


def get_blank_id(configs, symbol_table):
    if "ctc_conf" not in configs:
        configs["ctc_conf"] = {}

    if "<blank>" in symbol_table:
        if "ctc_blank_id" in configs["ctc_conf"]:
            assert configs["ctc_conf"]["ctc_blank_id"] == symbol_table["<blank>"]
        else:
            configs["ctc_conf"]["ctc_blank_id"] = symbol_table["<blank>"]
    else:
        assert "ctc_blank_id" in configs["ctc_conf"], "PLZ set ctc_blank_id in yaml"

    return configs, configs["ctc_conf"]["ctc_blank_id"]


def class2str(target, char_dict):
    content = []
    for w in target:
        content.append(char_dict[int(w)])
    return "".join(content).replace("▁", " ")


def milliseconds_to_hhmmssms(milliseconds):
    """
    Convert milliseconds to hh:mm:ss:ms format.

    Args:
        milliseconds (int): The total number of milliseconds.

    Returns:
        str: The formatted time string in hh:mm:ss:ms.
    """
    # Calculate hours, minutes, seconds, and remaining milliseconds
    hours = milliseconds // (1000 * 60 * 60)
    remaining_ms = milliseconds % (1000 * 60 * 60)
    minutes = remaining_ms // (1000 * 60)
    remaining_ms %= 1000 * 60
    seconds = remaining_ms // 1000
    remaining_ms %= 1000

    # Format the result
    return f"{hours:02}:{minutes:02}:{seconds:02}:{remaining_ms:03}"  # noqa: E231


def get_output(hyps, char_dict, model_type):
    decodes = []
    for hyp in hyps:
        if model_type == "asr_model":
            hyp = remove_duplicates_and_blank(hyp)
        decode = class2str(hyp, char_dict).strip()
        decodes.append(decode)
    return decodes


def get_output_with_timestamps(hyps, char_dict, model_type, max_silence_duration):
    decodes = []
    max_silence = max_silence_duration // 0.08  # 80ms per frame
    for tokens in hyps:  # cost O(input_batch_size | ccu)
        tokens = tokens.cpu()
        start = -1
        end = -1
        prev_end = -1
        silence_cum = 0
        decode_per_time = []
        decode = []
        for time_stamp in range(tokens.shape[0]):
            blk_mask = tokens[time_stamp] == 0
            if blk_mask.all():
                silence_cum += 1
            else:
                if (start == -1) and (end == -1):
                    if prev_end != -1:
                        start = max(math.ceil((time_stamp + prev_end) / 2), time_stamp - 2)
                    else:
                        start = max(time_stamp - 2, 0)
                silence_cum = 0

                decode_per_time.extend(tokens[time_stamp][~blk_mask].tolist())

            if (silence_cum == max_silence) and (start != -1):
                end = time_stamp
                prev_end = end
                item = {
                    "decode": get_output([decode_per_time], char_dict, model_type)[0],
                    "start": milliseconds_to_hhmmssms(start * 8 * 10),
                    "end": milliseconds_to_hhmmssms(end * 8 * 10),
                }
                decode.append(item)
                decode_per_time = []
                start = -1
                end = -1
                silence_cum = 0

        if (start != -1) and (end == -1) and (len(decode_per_time) > 0):
            item = {
                "decode": get_output([decode_per_time], char_dict, model_type)[0],
                "start": milliseconds_to_hhmmssms(start * 8 * 10),
                "end": milliseconds_to_hhmmssms(time_stamp * 8 * 10),
            }
            decode.append(item)
        decodes.append(decode)

    return decodes
