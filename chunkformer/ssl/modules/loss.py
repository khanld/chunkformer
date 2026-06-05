import torch


class MLMLoss:
    """Masked-language-model style negative log-likelihood loss used by BEST-RQ.

    The logits are expected to already be log-softmax normalised over the
    codebook classes; targets are the random-projection-quantizer codebook ids.
    """

    def __init__(self):
        super().__init__()
        self.nll_loss = torch.nn.NLLLoss()

    def __call__(self, logits, targets):
        loss = self.nll_loss(logits, targets)
        loss = torch.mean(loss)
        return loss
