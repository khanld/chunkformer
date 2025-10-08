import os
from functools import partial

from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy, transformer_auto_wrap_policy

from chunkformer.modules.decoder_layer import DecoderLayer
from chunkformer.modules.encoder_layer import ChunkFormerEncoderLayer
from chunkformer.utils.checkpoint import save_state_dict_and_infos
from chunkformer.utils.init_model import CHUNKFORMER_DECODER_CLASSES, CHUNKFORMER_ENCODER_CLASSES

CHUNKFORMER_ENCODER_LAYERS_CLASSES = {"chunkformer_encoder_layer": ChunkFormerEncoderLayer}

CHUNKFORMER_DECODER_LAYERS_CLASSES = {
    "transformer_decoder_layer": DecoderLayer,
    # TODO(Mddct):
    #     1 wrap transducer's predictor and joint
    #     2 wrap paraformer's cif and ignore lstm
}


def wenet_fsdp_wrap_policy(mode):
    # different wrap methods
    # please refer： https://openmmlab.medium.com/its-2023-is-pytorch-s-fsdp-the-best-choice-for-training-large-models-fe8d2848832f # noqa
    assert mode in ["no_shard", "model", "zero2", "zero3"]
    if mode == "no_shard":
        return None
    else:
        # TODO(Mddct):  Support user customization
        # see more wrap methods:
        # https://github.com/meta-llama/llama-recipes/blob/main/src/llama_recipes/utils/fsdp_utils.py#L13 # noqa
        if mode == "model":
            enc_dec_wrap_policy = partial(
                lambda_auto_wrap_policy,
                lambda_fn=lambda module: isinstance(
                    module,
                    tuple(CHUNKFORMER_ENCODER_CLASSES.values())
                    + tuple(CHUNKFORMER_DECODER_CLASSES.values()),
                ),
            )
            return enc_dec_wrap_policy
        else:
            to_wrap_class = set()
            to_wrap_class.update(set(CHUNKFORMER_ENCODER_LAYERS_CLASSES.values()))
            to_wrap_class.update(set(CHUNKFORMER_DECODER_LAYERS_CLASSES.values()))
            layers_wrap_policy = partial(
                transformer_auto_wrap_policy, transformer_layer_cls=to_wrap_class
            )
            return layers_wrap_policy


fullstate_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)


def fsdp_save_model(model, save_model_path, info_dict):
    # TODO(Mddct); When the model is large, saving a model will take a long time.
    # We only need to keep the sharding in an asynchronous manner, but it is
    # good now. This feature will be supported when llm is supported in the future.

    rank = int(os.environ.get("RANK", 0))
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, fullstate_save_policy):
        state_dict = model.state_dict()
        if rank == 0:
            save_state_dict_and_infos(state_dict, save_model_path, info_dict)


def check_gradient_checkpoint(model):
    ckpt_laye_types = []
    if hasattr(model, "encoder") and hasattr(model.encoder, "gradient_checkpointing"):
        if model.encoder.gradient_checkpointing:
            model.encoder.gradient_checkpointing = False
            ckpt_laye_types += list(CHUNKFORMER_ENCODER_LAYERS_CLASSES.values())
    if hasattr(model, "decoder") and hasattr(model.decoder, "gradient_checkpointing"):
        if model.decoder.gradient_checkpointing:
            model.decoder.gradient_checkpointing = False
            ckpt_laye_types += list(CHUNKFORMER_DECODER_LAYERS_CLASSES.values())
    return tuple(ckpt_laye_types)


def apply_fsdp_checkpointing(model, ckpt_layer_types: tuple):
    # NOTE(Mddct):  torch.utils.checkpoint is currently incompatible with
    # wenet's model mode. Using this writing method, Please refer to
    # https://github.com/meta-llama/llama-recipes/blob/main/src/llama_recipes/policies/activation_checkpointing_functions.py#L21 # noqa
    if len(ckpt_layer_types) == 0:
        return
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl,
        apply_activation_checkpointing,
        checkpoint_wrapper,
    )

    non_reentrant_wrapper = partial(
        checkpoint_wrapper,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=non_reentrant_wrapper,
        check_fn=lambda submodule: isinstance(submodule, ckpt_layer_types),
    )
