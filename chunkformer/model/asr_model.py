import torch

from .ctc import CTC
from .utils.common import IGNORE_ID

# from model.mwer import MWER


class ASRModel(torch.nn.Module):
    """CTC-attention hybrid Encoder-Decoder model"""

    def __init__(
        self,
        vocab_size: int,
        encoder,
        ctc: CTC,
        ctc_weight: float = 0.5,
        ignore_id: int = IGNORE_ID,
        reverse_weight: float = 0.0,
    ):
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight

        super().__init__()
        # note that eos is the same as sos (equivalent ID)
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.reverse_weight = reverse_weight

        self.encoder = encoder
        self.encoder.ctc = ctc
        self.ctc = ctc
