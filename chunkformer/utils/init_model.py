# Copyright (c) 2022 Binbin Zhang (binbzha@qq.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import torch

from ..modules.asr_model import ASRModel
from ..modules.cmvn import GlobalCMVN
from ..modules.ctc import CTC
from ..modules.decoder import BiTransformerDecoder, TransformerDecoder
from ..modules.encoder import ChunkFormerEncoder
from ..transducer.joint import TransducerJoint
from ..transducer.predictor import ConvPredictor, EmbeddingPredictor, RNNPredictor
from ..transducer.transducer import Transducer
from .checkpoint import load_checkpoint, load_trained_modules
from .cmvn import load_cmvn

CHUNKFORMER_ENCODER_CLASSES = {
    "chunkformer": ChunkFormerEncoder,
}

CHUNKFORMER_DECODER_CLASSES = {
    "transformer": TransformerDecoder,
    "bitransformer": BiTransformerDecoder,
}

CHUNKFORMER_CTC_CLASSES = {
    "ctc": CTC,
}

CHUNKFORMER_PREDICTOR_CLASSES = {
    "rnn": RNNPredictor,
    "embedding": EmbeddingPredictor,
    "conv": ConvPredictor,
}

CHUNKFORMER_JOINT_CLASSES = {
    "transducer_joint": TransducerJoint,
}

CHUNKFORMER_MODEL_CLASSES = {
    "asr_model": ASRModel,
    "transducer": Transducer,
}


def init_speech_model(args, configs):
    # Load global CMVN if specified
    if configs.get("cmvn", None) == "global_cmvn":
        mean, istd = load_cmvn(
            configs["cmvn_conf"]["cmvn_file"], configs["cmvn_conf"]["is_json_cmvn"]
        )
        global_cmvn = GlobalCMVN(torch.from_numpy(mean).float(), torch.from_numpy(istd).float())
    else:
        global_cmvn = None

    input_dim = configs["input_dim"]
    vocab_size = configs["output_dim"]

    # ChunkFormer only supports chunkformer encoder
    encoder_type = configs.get("encoder", "chunkformer")
    decoder_type = configs.get("decoder", "transformer")
    ctc_type = configs.get("ctc", "ctc")

    # Create ChunkFormer encoder
    encoder = CHUNKFORMER_ENCODER_CLASSES[encoder_type](
        input_dim, global_cmvn=global_cmvn, **configs["encoder_conf"]
    )

    # Create decoder
    decoder = CHUNKFORMER_DECODER_CLASSES[decoder_type](
        vocab_size, encoder.output_size(), **configs["decoder_conf"]
    )

    # Create CTC
    ctc = CHUNKFORMER_CTC_CLASSES[ctc_type](
        vocab_size,
        encoder.output_size(),
        blank_id=configs["ctc_conf"]["ctc_blank_id"] if "ctc_conf" in configs else 0,
    )

    # Create model based on type
    model_type = configs.get("model", "asr_model")
    if model_type == "transducer":
        predictor_type = configs.get("predictor", "rnn")
        joint_type = configs.get("joint", "transducer_joint")
        predictor = CHUNKFORMER_PREDICTOR_CLASSES[predictor_type](
            vocab_size, **configs["predictor_conf"]
        )
        joint = CHUNKFORMER_JOINT_CLASSES[joint_type](vocab_size, **configs["joint_conf"])
        model = CHUNKFORMER_MODEL_CLASSES[model_type](
            vocab_size=vocab_size,
            blank=0,
            predictor=predictor,
            encoder=encoder,
            attention_decoder=decoder,
            joint=joint,
            ctc=ctc,
            special_tokens=configs.get("tokenizer_conf", {}).get("special_tokens", None),
            **configs["model_conf"],
        )
    else:
        model = CHUNKFORMER_MODEL_CLASSES[model_type](
            vocab_size=vocab_size,
            encoder=encoder,
            decoder=decoder,
            ctc=ctc,
            special_tokens=configs.get("tokenizer_conf", {}).get("special_tokens", None),
            **configs["model_conf"],
        )
    return model, configs


def init_model(args, configs):
    """Initialize ChunkFormer model"""
    model_type = configs.get("model", "asr_model")
    configs["model"] = model_type
    model, configs = init_speech_model(args, configs)

    # Load checkpoint if specified
    if hasattr(args, "checkpoint") and args.checkpoint is not None:
        infos = load_checkpoint(model, args.checkpoint)
    elif hasattr(args, "enc_init") and args.enc_init is not None:
        infos = load_trained_modules(model, args)
    else:
        infos = {}
    configs["init_infos"] = infos

    # Tie weights if model supports it
    if hasattr(model, "tie_or_clone_weights"):
        jit = not hasattr(args, "jit") or args.jit
        model.tie_or_clone_weights(jit)

    if int(os.environ.get("RANK", 0)) == 0:
        print(configs)

    return model, configs
