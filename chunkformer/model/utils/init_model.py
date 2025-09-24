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

from ..asr_model import ASRModel
from ..cmvn import GlobalCMVN
from ..ctc import CTC
from ..encoder import ChunkFormerEncoder
from .cmvn import load_cmvn


def init_model(configs, config_path):
    if configs["cmvn_file"] is not None:
        cmvn_file = os.path.abspath(os.path.join(config_path, "..", "..", configs["cmvn_file"]))
        mean, istd = load_cmvn(cmvn_file, configs["is_json_cmvn"])
        global_cmvn = GlobalCMVN(torch.from_numpy(mean).float(), torch.from_numpy(istd).float())
    else:
        global_cmvn = None

    input_dim = configs["input_dim"]
    vocab_size = configs["output_dim"]

    encoder = ChunkFormerEncoder(input_dim, global_cmvn=global_cmvn, **configs["encoder_conf"])

    ctc = CTC(vocab_size, encoder._output_size)

    model = ASRModel(vocab_size=vocab_size, encoder=encoder, ctc=ctc)

    return model
