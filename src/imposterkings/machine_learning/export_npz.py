"""Export a trained ``.pt`` checkpoint to a torch-free ``.npz`` -- the format the shipped game reads.

    python -m imposterkings.machine_learning.export_npz models/gen1_v3c_v2feat/attn_d64_L2.pt models/mlp_256.pt

A ``.pt`` is a torch pickle: reading one requires torch (4.2 GB) even though inference itself is a handful
of matmuls. So the release ships ``.npz`` -- a plain zip of numpy arrays -- and ``npz_infer`` runs the
forward pass. This exporter is a DEV tool (it needs torch); the game never calls it.

Weight names are preserved verbatim from the state dict for the attention net, so ``npz_infer`` indexes
them by the same keys the torch module uses. The MLP's ``nn.Sequential`` indices are renumbered to a dense
``l0/l1/...`` (the Sequential holds ReLUs too, so its raw indices are gappy). Config travels in
``__dunder__`` keys, which the loader separates from the weights.
"""
from __future__ import annotations

import argparse
import os
from typing import Dict

import numpy as np
import torch


def export(src: str, dst: str = None) -> str:
    """``src`` .pt -> ``dst`` .npz (default: same path with the suffix swapped). Returns the written path."""
    dst = dst or os.path.splitext(src)[0] + ".npz"
    blob = torch.load(src, map_location="cpu", weights_only=False)
    sd = blob["state_dict"]
    out: Dict[str, np.ndarray] = {}

    if blob.get("model_type") == "attention":
        cfg = blob["config"]
        for k, v in sd.items():
            out[k] = v.detach().cpu().numpy().astype(np.float32)     # keys verbatim: npz_infer indexes them
        out["__model_type__"] = np.array("attention")
        for key in ("d_model", "n_layers", "n_heads", "ffn_hidden", "bounded", "feat"):
            out[f"__{key}__"] = np.array(cfg[key])
    else:                                                            # MLP: dense-renumber the Sequential
        linears = [k[:-len(".weight")] for k in sd if k.endswith(".weight")]
        linears.sort(key=lambda p: int(p.split(".")[-1]))            # net.0, net.2, net.4 ... -> l0, l1, l2
        for i, p in enumerate(linears):
            out[f"l{i}.weight"] = sd[f"{p}.weight"].detach().cpu().numpy().astype(np.float32)
            out[f"l{i}.bias"] = sd[f"{p}.bias"].detach().cpu().numpy().astype(np.float32)
        out["__model_type__"] = np.array("mlp")
        out["__feature_dim__"] = np.array(blob["feature_dim"])
        out["__arch__"] = np.array(blob["arch"])

    np.savez_compressed(dst, **out)
    n = sum(v.size for k, v in out.items() if not k.startswith("__"))
    print(f"{src} -> {dst}  ({n:,} params, {os.path.getsize(dst)/1024:.0f} KB)")
    return dst


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Export .pt checkpoints to torch-free .npz for the release.")
    p.add_argument("checkpoints", nargs="+", help="one or more .pt files")
    p.add_argument("--out-dir", default=None, help="write the .npz here (default: alongside the .pt)")
    a = p.parse_args(argv)
    for ck in a.checkpoints:
        dst = None
        if a.out_dir:
            os.makedirs(a.out_dir, exist_ok=True)
            dst = os.path.join(a.out_dir, os.path.splitext(os.path.basename(ck))[0] + ".npz")
        export(ck, dst)


if __name__ == "__main__":
    main()
