#!/usr/bin/env python
"""Minimal inference demo for Generative Modeling via Drifting.

Generates ImageNet class-conditional samples (1 NFE, one-step) using a
pretrained pixel-space model downloaded automatically from HuggingFace.
No ImageNet dataset, FID stats, or TPU required.

Usage:
    JAX_PLATFORMS=cpu python demo.py
    JAX_PLATFORMS=cpu python demo.py --classes 95,22,88 --cfg-scale 1.2
"""
import os
import sys
import argparse
from functools import partial
from pathlib import Path

# Set backend before importing JAX
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("HF_ROOT", str(Path(__file__).parent / "hf_cache"))

sys.path.insert(0, str(Path(__file__).parent))

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from utils.misc import prepare_rng
from utils.hsdp_util import set_global_mesh
from utils.init_util import load_generator_model_and_params
from utils.env import HF_ROOT


IMAGENET_LABELS = {
    95: "jacamar",
    22: "bald_eagle",
    88: "macaw",
    108: "sea_anemone",
    386: "african_elephant",
    296: "ice_bear",
    483: "castle",
    698: "palace",
}


def _is_latent(metadata: dict) -> bool:
    return metadata.get("model_config", {}).get("in_channels", 3) == 4


def _build_postprocess_fn(latent: bool):
    """Return postprocess function: model output -> float32 BCHW in [0,1]."""
    if latent:
        # Latent model: needs VAE decode (requires TPU in this codebase)
        from dataset.vae import vae_enc_decode
        _, decode_fn = vae_enc_decode(replicate_params=False)
        def postprocess(images):
            return jnp.clip((decode_fn(images) + 1) / 2, 0, 1)
        return postprocess
    # Pixel model: simple renormalization [-1,1] -> [0,1], BHWC -> BCHW
    def postprocess(images):
        return jnp.clip((images + 1) / 2, 0, 1).transpose(0, 3, 1, 2)
    return postprocess


def generate_step(batch, params, rng, apply_fn, postprocess_fn, cfg_scale=1.0):
    """Single forward pass: labels -> postprocessed images."""
    _, labels = batch
    samples = apply_fn(
        {"params": params},
        train=False,
        rngs=prepare_rng(rng, ["noise"]),
        c=labels,
        cfg_scale=cfg_scale,
    )["samples"]
    return postprocess_fn(samples)


def main():
    parser = argparse.ArgumentParser(description="Drifting model inference demo")
    parser.add_argument(
        "--model", default="hf://pixel_B_sota",
        help="hf://<name> or local artifact path. Default: hf://pixel_B_sota"
    )
    parser.add_argument(
        "--classes", default="95,22,88,108,386,296",
        help="Comma-separated ImageNet class IDs (0-999)"
    )
    parser.add_argument("--cfg-scale", type=float, default=1.0,
                        help="Classifier-free guidance scale (default: 1.0)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="demo_output",
                        help="Directory for saved images")
    args = parser.parse_args()

    class_ids = [int(c.strip()) for c in args.classes.split(",") if c.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize a 1x1 mesh for single-device execution
    set_global_mesh(1)

    print(f"Backend  : {jax.default_backend()}")
    print(f"Devices  : {jax.devices()}")
    print(f"Model    : {args.model}")
    print(f"Classes  : {class_ids}")
    print(f"CFG      : {args.cfg_scale}")
    print()

    print("Downloading / loading model weights from HuggingFace...")
    model, params, metadata = load_generator_model_and_params(
        args.model, hf_cache_dir=HF_ROOT
    )
    cfg = metadata.get("model_config", {})
    print(f"  hidden={cfg.get('hidden_size')} depth={cfg.get('depth')} "
          f"heads={cfg.get('num_heads')} patch={cfg.get('patch_size')} "
          f"latent={_is_latent(metadata)}")
    print()

    postprocess_fn = _build_postprocess_fn(_is_latent(metadata))
    gen_jit = jax.jit(
        partial(generate_step, apply_fn=model.apply, postprocess_fn=postprocess_fn)
    )

    # Warm up JIT compilation with the first sample
    print(f"Generating {len(class_ids)} images (JIT compile on first call)...")
    saved = []
    for i, class_id in enumerate(class_ids):
        label = np.array([class_id], dtype=np.int32)
        batch = (np.zeros((1, 1), dtype=np.int32), label)
        rng = jax.random.PRNGKey(args.seed + i)

        tag = IMAGENET_LABELS.get(class_id, f"cls{class_id}")
        print(f"  [{i+1:2d}/{len(class_ids)}] class {class_id:4d} ({tag})", end="", flush=True)

        out = gen_jit(batch, params=params, cfg_scale=args.cfg_scale, rng=rng)
        jax.block_until_ready(out)

        # out: BCHW float32 in [0,1]
        img_np = np.asarray(jax.device_get(out[0]), dtype=np.float32)  # CHW
        if img_np.ndim == 3 and img_np.shape[0] in (3, 4):
            img_np = img_np.transpose(1, 2, 0)            # HWC
        img_u8 = (np.clip(img_np[..., :3], 0, 1) * 255).astype(np.uint8)

        fname = output_dir / f"class_{class_id:03d}_{tag}.png" 
        Image.fromarray(img_u8).save(fname)
        print(f"  -> {fname}")
        saved.append((class_id, tag, img_u8))

    # Save a summary grid
    n = len(saved)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    H, W = saved[0][2].shape[:2]
    grid = np.zeros((rows * H, cols * W, 3), dtype=np.uint8)
    for idx, (_, _, img) in enumerate(saved):
        r, c = idx // cols, idx % cols
        grid[r * H:(r + 1) * H, c * W:(c + 1) * W] = img
    grid_path = output_dir / "grid.png"
    Image.fromarray(grid).save(grid_path)

    print(f"\nDone. {n} images + grid saved to: {output_dir}/")


if __name__ == "__main__":
    main()
