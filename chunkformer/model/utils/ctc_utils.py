import math

from .common import remove_duplicates_and_blank


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
    return f"{hours:02}:{minutes:02}:{seconds:02}:{remaining_ms:03}"


def get_output(hyps, char_dict):
    decodes = []
    for hyp in hyps:
        hyp = remove_duplicates_and_blank(hyp)
        decode = class2str(hyp, char_dict)
        decodes.append(decode)
    return decodes


def get_output_with_timestamps(hyps, char_dict):
    decodes = []
    max_silence = 20
    for tokens in hyps:  # cost O(input_batch_size | ccu)
        tokens = tokens.cpu()
        start = -1
        end = -1
        prev_end = -1
        silence_cum = 0
        decode_per_time = []
        decode = []
        for time_stamp, token in enumerate(tokens):
            if token == 0:
                silence_cum += 1
            else:
                if (start == -1) and (end == -1):
                    if prev_end != -1:
                        start = math.ceil((time_stamp + prev_end) / 2)
                    else:
                        start = max(time_stamp - int(max_silence / 2), 0)
                silence_cum = 0
                decode_per_time.append(token)

            if (silence_cum == max_silence) and (start != -1):
                end = time_stamp
                prev_end = end
                item = {
                    "decode": class2str(remove_duplicates_and_blank(decode_per_time), char_dict),
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
                "decode": class2str(remove_duplicates_and_blank(decode_per_time), char_dict),
                "start": milliseconds_to_hhmmssms(start * 8 * 10),
                "end": milliseconds_to_hhmmssms(time_stamp * 8 * 10),
            }
            decode.append(item)
        decodes.append(decode)

    return decodes
