"""Check inference primitives on an explicit device, without loading a policy."""

import argparse
import json

import torch
from torch import nn
from torch.nn import functional as F


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.device.startswith("musa"):
        import torch_musa  # noqa: F401
        if not torch.musa.is_available():
            raise RuntimeError("MUSA requested but unavailable; no CPU fallback")
    device = torch.device(args.device)
    if device.type not in ("cpu", "musa"):
        parser.error("Supported devices: cpu or musa[:index]")
    torch.set_num_threads(1)
    torch.manual_seed(0)
    projection = nn.Linear(32, 32).to(device).eval().requires_grad_(False)
    with torch.no_grad():
        x = torch.randn(1, 4, 8, 32, device=device)
        q = F.layer_norm(projection(x), (32,))
        output = F.scaled_dot_product_attention(q, q, q, dropout_p=0.0)
    if device.type == "musa":
        torch.musa.synchronize(device)
    if output.shape != x.shape or not torch.isfinite(output).all().item():
        raise FloatingPointError("Invalid output from inference operator check")
    print(json.dumps({"stage": "inference_primitives", "status": "passed",
                      "device": str(output.device), "torch": str(torch.__version__),
                      "operators": ["linear", "layer_norm", "scaled_dot_product_attention"],
                      "ardy_sonic_checked": False, "physics_executed": False,
                      "task_success": None}, indent=2))


if __name__ == "__main__":
    main()
