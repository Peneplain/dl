"""Explicit device selection for frozen inference. No automatic fallback."""

def device_for(name):
    import torch
    if name.startswith("musa"):
        import torch_musa  # noqa: F401
        if not torch.musa.is_available():
            raise RuntimeError("MUSA requested but unavailable; no automatic CPU fallback")
    device = torch.device(name)
    if device.type not in ("cpu", "musa"):
        raise ValueError("This bring-up entry supports cpu or musa[:index]")
    torch.zeros(1, device=device)
    return device


def synchronize(device):
    if device.type == "musa":
        import torch
        torch.musa.synchronize(device)


