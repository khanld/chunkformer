from typing import List

import torch


def optimized_search(
    model: torch.nn.Module,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    n_steps: int = 64,
) -> torch.Tensor:
    batch_size = encoder_out.size(0)
    max_len = encoder_out.size(1)
    # n_layers x B x F
    cache_m, cache_c = model.predictor.init_state(
        batch_size, method="zero", device=encoder_out.device
    )
    pred_input = (
        torch.tensor([model.blank] * batch_size).reshape(batch_size, 1).to(encoder_out.device)
    )
    # B, T, n_steps
    output = torch.zeros(batch_size, max_len * n_steps + 1, dtype=torch.int64).to(
        encoder_out.device
    )
    for t in range(max_len):
        valid_time_index = t < encoder_out_lens
        encoder_out_t = encoder_out[:, t : t + 1, :]  # [B, 1, E]

        for step in range(1, n_steps + 1):
            if step == 1:
                non_blank_mask = torch.ones(batch_size, device=encoder_out.device).bool()
            else:
                non_blank_mask = (output[:, t * n_steps + step - 1] != model.blank).squeeze()  # [B]

            non_blank_mask = non_blank_mask & valid_time_index  # [B]
            if non_blank_mask.sum() == 0:
                break

            encoder_out_t_step = encoder_out_t[non_blank_mask]  # [B', 1, E]
            pred_input_step = pred_input[non_blank_mask]  # [B', 1]
            cache_m_no_blk, cache_c_no_blk = (
                cache_m[:, non_blank_mask],
                cache_c[:, non_blank_mask],
            )  # [2, n_layers, B', F]

            pred_out_step, new_cache = model.predictor.forward_step(
                pred_input_step, (cache_m_no_blk, cache_c_no_blk)
            )  # [B, 1, P]

            joint_out_step = model.joint(encoder_out_t_step, pred_out_step)  # [B, 1, V]
            joint_out_probs = joint_out_step.log_softmax(dim=-1)

            joint_out_max = joint_out_probs.argmax(dim=-1).squeeze()  # [B]
            output[non_blank_mask, t * n_steps + step] = joint_out_max

            # update cache
            pred_input[non_blank_mask] = torch.where(
                (joint_out_max != model.blank).reshape(-1, 1),
                joint_out_max.reshape(-1, 1),
                pred_input[non_blank_mask],
            )
            cache_m[:, non_blank_mask] = torch.where(
                (joint_out_max != model.blank).reshape(1, -1, 1),
                new_cache[0],
                cache_m[:, non_blank_mask],
            )
            cache_c[:, non_blank_mask] = torch.where(
                (joint_out_max != model.blank).reshape(1, -1, 1),
                new_cache[1],
                cache_c[:, non_blank_mask],
            )

    output = output[:, 1:]  # remove sos
    return output


# Buy me a coffee, gray matters
def batch_greedy_search(
    model: torch.nn.Module,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    n_steps: int = 64,
) -> List[List[int]]:
    output = optimized_search(model, encoder_out, encoder_out_lens, n_steps)
    no_blk_output = output != model.blank
    output_lens = no_blk_output.sum(dim=1)

    output = output[no_blk_output]
    hyps = torch.split(output, output_lens.tolist())

    result: List[List[int]] = [hyp.tolist() for hyp in hyps]
    return result


def basic_greedy_search(
    model: torch.nn.Module,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    n_steps: int = 64,
) -> List[List[int]]:
    # sos
    pred_input_step = torch.tensor([model.blank]).reshape(1, 1).to(encoder_out.device)
    cache = model.predictor.init_state(1, method="zero", device=encoder_out.device)
    new_cache: List[torch.Tensor] = []
    t = 0
    hyps = []
    prev_out_nblk = True
    pred_out_step = None
    per_frame_max_noblk = n_steps
    per_frame_noblk = 0
    while t < encoder_out_lens:
        encoder_out_step = encoder_out[:, t : t + 1, :]  # [1, 1, E]
        if prev_out_nblk:
            step_outs = model.predictor.forward_step(pred_input_step, cache)  # [1, 1, P]
            pred_out_step, new_cache = step_outs[0], step_outs[1]

        joint_out_step = model.joint(encoder_out_step, pred_out_step)  # [1,1,v]
        joint_out_probs = joint_out_step.log_softmax(dim=-1)

        joint_out_max = joint_out_probs.argmax(dim=-1).squeeze()  # []
        if joint_out_max != model.blank:
            hyps.append(joint_out_max.item())
            prev_out_nblk = True
            per_frame_noblk = per_frame_noblk + 1
            pred_input_step = joint_out_max.reshape(1, 1)
            # state_m, state_c =  clstate_out_m, state_out_c
            cache = new_cache

        if joint_out_max == model.blank or per_frame_noblk >= per_frame_max_noblk:
            if joint_out_max == model.blank:
                prev_out_nblk = False
            # TODO(Mddct): make t in chunk for streamming
            # or t should't be too lang to predict none blank
            t = t + 1
            per_frame_noblk = 0

    return [hyps]


def greedy_search(
    model: torch.nn.Module,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    n_steps: int = 64,
) -> List[List[int]]:
    """
    Unified greedy search interface that automatically selects the best implementation.

    Args:
        model: Transducer model with predictor and joint networks
        encoder_out: Encoder outputs [B, T, E] where B is batch size
        encoder_out_lens: Length of each sequence in batch [B]
        n_steps: Maximum non-blank predictions per frame

    Returns:
        List of hypothesis lists, one for each sequence in batch
    """
    batch_size = encoder_out.size(0)
    if batch_size == 1:
        # For single sequences, use the basic implementation
        return basic_greedy_search(model, encoder_out, encoder_out_lens, n_steps)
    else:
        return batch_greedy_search(model, encoder_out, encoder_out_lens, n_steps)
