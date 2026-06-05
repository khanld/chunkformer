import torch


class MLMLoss:
    """Masked-language-model style negative log-likelihood loss used by BEST-RQ.

    The logits are expected to already be log-softmax normalised over the
    codebook classes; targets are the random-projection-quantizer codebook ids.
    """

    def __init__(
        self,
        mask_threshold: float = 0.8,
    ):
        super().__init__()
        self.nll_loss = torch.nn.NLLLoss()
        self.mask_threshold = mask_threshold

    def __call__(self, logits, targets):
        loss = self.nll_loss(logits, targets)
        loss = torch.mean(loss)
        return loss
