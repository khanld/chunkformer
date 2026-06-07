# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang)
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

import datetime
import logging
import os
import re
from collections import OrderedDict
from typing import Any

import torch
import yaml


def resolve_checkpoint_path(path: str, filename: str = "pytorch_model.pt") -> str:
    """Resolve a checkpoint location to a local ``.pt`` file path.

    Accepts, in order of priority:

    1. A local ``.pt`` file path (returned unchanged).
    2. A local directory that contains ``filename`` (returns the file inside it).
    3. A Hugging Face Hub repo id, e.g. ``khanhld/vip-vl-base-vie``. In this case
       ``filename`` is downloaded from the Hub (cached locally) and the cached
       path is returned. Authentication for private repos is taken from the
       ``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN`` environment variable or a prior
       ``huggingface-cli login``.

    This keeps full backward compatibility: existing local paths are returned
    as-is and only non-existent paths are treated as Hub repo ids.
    """
    # 1. Local checkpoint file.
    if os.path.isfile(path):
        return path

    # 2. Local directory holding the checkpoint file.
    if os.path.isdir(path):
        candidate = os.path.join(path, filename)
        if os.path.isfile(candidate):
            return candidate
        raise FileNotFoundError(f"Checkpoint directory '{path}' does not contain '{filename}'.")

    # 3. Otherwise treat the string as a Hugging Face Hub repo id.
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:  # pragma: no cover - optional dependency
        raise ImportError(
            f"'{path}' is not a local file/directory and 'huggingface_hub' is not "
            "installed, so it cannot be downloaded from the Hugging Face Hub. "
            "Install it with `pip install huggingface_hub` or pass a local path."
        ) from e

    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        logging.info(
            "Checkpoint: '%s' is not a local path; downloading '%s' from the " "Hugging Face Hub.",
            path,
            filename,
        )
    return str(hf_hub_download(repo_id=path, filename=filename))


def load_checkpoint(model: torch.nn.Module, path: str) -> dict:
    rank = int(os.environ.get("RANK", 0))
    path = resolve_checkpoint_path(path)
    logging.info("[Rank {}] Checkpoint: loading from checkpoint {}".format(rank, path))
    checkpoint = torch.load(path, map_location="cpu", mmap=True)
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
    if rank == 0:
        for key in missing_keys:
            logging.info("missing tensor: {}".format(key))
        for key in unexpected_keys:
            logging.info("unexpected tensor: {}".format(key))
    info_path = re.sub(".pt$", ".yaml", path)
    configs = {}
    if os.path.exists(info_path) and info_path.endswith(".yaml"):
        with open(info_path, "r") as fin:
            configs = yaml.load(fin, Loader=yaml.FullLoader)
    return configs


def save_state_dict_and_infos(state_dict, path: str, infos=None):
    rank = int(os.environ.get("RANK", 0))
    logging.info("[Rank {}] Checkpoint: save to checkpoint {}".format(rank, path))
    torch.save(state_dict, path)
    info_path = re.sub(".pt$", ".yaml", path)
    if infos is None:
        infos = {}
    infos["save_time"] = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    with open(info_path, "w") as fout:
        data = yaml.dump(infos)
        fout.write(data)


def save_checkpoint(model: torch.nn.Module, path: str, infos=None):
    """
    Args:
        infos (dict or None): any info you want to save.
    """
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    elif isinstance(model, torch.nn.parallel.DistributedDataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    save_state_dict_and_infos(state_dict, path, infos)


def filter_modules(model_state_dict, modules):
    rank = int(os.environ.get("RANK", 0))
    new_mods = []
    incorrect_mods = []
    mods_model = model_state_dict.keys()
    for mod in modules:
        if any(key.startswith(mod) for key in mods_model):
            new_mods += [mod]
        else:
            incorrect_mods += [mod]
    if incorrect_mods and rank == 0:
        logging.warning(
            "module(s) %s don't match or (partially match) " "available modules in model.",
            incorrect_mods,
        )
        logging.warning("for information, the existing modules in model are:")
        logging.warning("%s", mods_model)

    return new_mods


def load_trained_modules(model: torch.nn.Module, args: Any):
    # Load encoder modules with pre-trained model(s).
    enc_model_path = resolve_checkpoint_path(args.enc_init)
    enc_modules = args.enc_init_mods
    main_state_dict = model.state_dict()
    logging.warning("model(s) found for pre-initialization")
    if os.path.isfile(enc_model_path):
        logging.info("Checkpoint: loading from checkpoint %s for CPU" % enc_model_path)
        model_state_dict = torch.load(enc_model_path, map_location="cpu")
        modules = filter_modules(model_state_dict, enc_modules)
        partial_state_dict = OrderedDict()
        for key, value in model_state_dict.items():
            if any(key.startswith(m) for m in modules):
                partial_state_dict[key] = value
        main_state_dict.update(partial_state_dict)
    else:
        logging.warning("model was not found : %s", enc_model_path)

    model.load_state_dict(main_state_dict)
    configs: dict = {}
    return configs
