import torch


def load_checkpoint(model: torch.nn.Module, path: str):
    if torch.cuda.is_available():
        print("Checkpoint: loading from checkpoint %s for GPU" % path)
        checkpoint = torch.load(path, weights_only=True)
    else:
        print("Checkpoint: loading from checkpoint %s for CPU" % path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint, strict=False)
