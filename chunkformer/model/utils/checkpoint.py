import logging
import os
import re

import yaml
import torch
from collections import OrderedDict

import datetime


def load_checkpoint(model: torch.nn.Module, path: str) -> dict:
    if torch.cuda.is_available():
        print('Checkpoint: loading from checkpoint %s for GPU' % path)
        checkpoint = torch.load(path, weights_only=True)
    else:
        print('Checkpoint: loading from checkpoint %s for CPU' % path)
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
    # print(f"Missing keys: {missing_keys}")
    # print(f"Unexpected keys: {unexpected_keys}")