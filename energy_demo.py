#!/usr/bin/env python
"""Energy-guided generation for Generative Modeling via Drifting.

Compares three generation strategies for the same class and noise seed:

  baseline:  standard 1-NFE generation (no guidance)
  method 1:  direct output guidance (L∞-normalised gradient)
               x' = clip(x - α₁ · ∇_x E(x) / ‖∇_x E(x)‖∞, -1, 1)
               α₁ = --alpha-m1 (default 0.3)
               L∞ normalisation makes α₁ the exact per-pixel shift for the
               most-affected channel (e.g. α₁=0.3 → R pixels shift by 0.3)
  method 2:  noise-space guidance (back-propagates through the model)
               for t in range(n_steps):
                 ε = ε - α₂ · ∇_ε E(f(ε)) / ‖∇_ε E(f(ε))‖₂
               x' = f(ε)
               α₂ = --alpha (default 0.3)
               L2 normalisation over ε gives a consistent step size
               regardless of model Jacobian scale.

Energy options
--------------
  red    prefer images with high R relative to G and B
  blue   prefer images with high B relative to R and G
  green  prefer images with high G relative to R and B
  bright prefer overall bright images
  dark   prefer overall dark images

Usage
-----
  JAX_PLATFORMS=cpu python energy_demo.py
  JAX_PLATFORMS=cpu python energy_demo.py --energy blue --alpha 0.5 --steps 3
  JAX_PLATFORMS=cpu python energy_demo.py --classes 95,22,88 --energy red --alpha 0.3
"""
import os
import sys
import argparse
from functools import partial
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("HF_ROOT", str(Path(__file__).parent / "hf_cache"))
sys.path.insert(0, str(Path(__file__).parent))

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from utils.hsdp_util import set_global_mesh
from utils.init_util import load_generator_model_and_params
from utils.env import HF_ROOT
from models.generator import DitGen


# ---------------------------------------------------------------------------
# Energy functions
# ---------------------------------------------------------------------------
# Convention: E(x) is *minimised*.  x is BHWC float32 in [-1, 1].
# The gradient step x - α·∇E moves x toward lower energy (preferred region).

def energy_red(x):
    """Prefer reddish images (high R, low G and B)."""
    x = x.astype(jnp.float32)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    return -jnp.mean(r - 0.5 * g - 0.5 * b)

def energy_blue(x):
    """Prefer bluish images."""
    x = x.astype(jnp.float32)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    return -jnp.mean(b - 0.5 * r - 0.5 * g)

def energy_green(x):
    """Prefer greenish images."""
    x = x.astype(jnp.float32)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    return -jnp.mean(g - 0.5 * r - 0.5 * b)

def energy_bright(x):
    """Prefer bright images."""
    return -jnp.mean(x.astype(jnp.float32))

def energy_dark(x):
    """Prefer dark images."""
    return jnp.mean(x.astype(jnp.float32))

ENERGIES = {
    "red":    energy_red,
    "blue":   energy_blue,
    "green":  energy_green,
    "bright": energy_bright,
    "dark":   energy_dark,
}


# ---------------------------------------------------------------------------
# Model forward helpers
# ---------------------------------------------------------------------------

def _get_cond(model, params, labels, cfg_scale, rng_labels, noise_coords, noise_classes):
    """Compute conditioning vector (deterministic once noise_labels are fixed)."""
    B = labels.shape[0]
    noise_labels = jax.random.randint(
        rng_labels, (B, max(1, noise_coords)), 0, max(1, noise_classes)
    )
    return model.apply(
        {"params": params},
        labels, cfg_scale, noise_labels,
        method=DitGen.c_cfg_noise_to_cond,
    ), noise_labels


def _forward(model, params, epsilon, cond):
    """Pure spatial forward pass: (epsilon BHWC, cond) -> samples BHWC [-1,1]."""
    return model.apply({"params": params}, epsilon, cond,
                       method=DitGen.generate_image)


# ---------------------------------------------------------------------------
# Guidance methods
# ---------------------------------------------------------------------------

def method1_output(x_base, energy_fn, alpha):
    """
    Method 1 – direct output guidance.

    Computes the gradient of E w.r.t. the output pixels x, then takes one
    gradient-descent step using L∞ normalisation.  L∞ makes alpha directly
    interpretable: the most-affected pixel shifts by exactly alpha on the
    [-1, 1] scale.  For a uniform energy like "redness", every pixel in
    the dominant channel shifts by alpha.

    The result is clipped to [-1, 1].  No model re-evaluation is needed.
    """
    grad = jax.grad(energy_fn)(x_base.astype(jnp.float32))
    g_max = jnp.max(jnp.abs(grad)) + 1e-8  # L∞ normalisation
    x_new = x_base.astype(jnp.float32) - alpha * grad / g_max
    return jnp.clip(x_new, -1.0, 1.0)


def method2_noise(model, params, epsilon, cond, energy_fn, alpha, n_steps):
    """
    Method 2 – noise-space guidance.

    Back-propagates the energy gradient through the full model to update the
    input noise ε.  Each step requires one forward + one backward pass.
    Because ε stays on (or near) the prior N(0,1), the final sample is more
    likely to lie on the learned image manifold.

    Uses jax.lax.scan so the n_steps loop is compiled into a single XLA
    program with shared activations across steps.
    """
    def one_step(eps, _):
        def E_of_eps(e):
            x = _forward(model, params, e, cond)
            return energy_fn(x.astype(jnp.float32))
        grad = jax.grad(E_of_eps)(eps.astype(jnp.float32))
        g_norm = jnp.linalg.norm(grad) + 1e-8
        eps_new = eps.astype(jnp.float32) - alpha * grad / g_norm
        return eps_new, None

    eps_guided, _ = jax.lax.scan(one_step, epsilon.astype(jnp.float32), None, length=n_steps)
    x_guided = _forward(model, params, eps_guided, cond)
    return x_guided, eps_guided


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

IMAGENET_LABELS = {
    95: "jacamar", 22: "bald_eagle", 88: "macaw",
    108: "sea_anemone", 386: "african_elephant", 296: "ice_bear",
    483: "castle", 698: "palace",
}


def _to_uint8(x_bhwc):
    """First image in batch: BHWC [-1,1] -> HWC uint8 [0,255]."""
    img = np.asarray(jax.device_get(x_bhwc[0])).astype(np.float32)
    img = np.clip((img + 1.0) / 2.0, 0.0, 1.0)
    return (img[..., :3] * 255).astype(np.uint8)


def _channel_stats(x_bhwc):
    """Mean R/G/B for the first image (in [0,1] range)."""
    img = np.asarray(jax.device_get(x_bhwc[0])).astype(np.float32)
    img = np.clip((img + 1.0) / 2.0, 0.0, 1.0)
    r, g, b = img[..., 0].mean(), img[..., 1].mean(), img[..., 2].mean()
    return r, g, b


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Energy-guided Drifting demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
--alpha-m1  (method 1, L∞-normalised): fraction of pixel range [-1,1] shifted
  0.1  subtle colour tint
  0.3  clearly visible colour shift
  0.5  strong colour bias, may look unnatural
  0.9  near-maximum shift, extreme colour distortion

--alpha     (method 2, L2-normalised): magnitude of noise update relative to noise std
  0.1  subtle
  0.3  moderate, recommended default
  0.5  strong, few steps sufficient
        """,
    )
    ap.add_argument("--model", default="hf://pixel_B_sota")
    ap.add_argument("--classes", default="95,22,88",
                    help="Comma-separated ImageNet class IDs (default: 95,22,88)")
    ap.add_argument("--energy", default="red", choices=list(ENERGIES),
                    help="Energy type (default: red)")
    ap.add_argument("--alpha", type=float, default=0.3,
                    help="Step size for method 2 – noise space, L2-normalised (default: 0.3)")
    ap.add_argument("--alpha-m1", type=float, default=0.3,
                    help="Step size for method 1 – output space, L∞-normalised (default: 0.3)")
    ap.add_argument("--steps", type=int, default=3,
                    help="Gradient steps for method 2 – noise space (default: 3)")
    ap.add_argument("--cfg-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", default="energy_output")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_global_mesh(1)

    print(f"Backend  : {jax.default_backend()}")
    print(f"Devices  : {jax.devices()}")
    print(f"Model    : {args.model}")
    print(f"Energy   : {args.energy}  alpha-m1={args.alpha_m1}  alpha-m2={args.alpha}  steps={args.steps}")
    print()

    print("Loading model weights...")
    model, params, metadata = load_generator_model_and_params(
        args.model, hf_cache_dir=HF_ROOT
    )
    cfg = metadata.get("model_config", {})
    in_channels  = cfg.get("in_channels",   3)
    input_size   = cfg.get("input_size",   256)
    noise_coords = cfg.get("noise_coords",   1)
    noise_classes = cfg.get("noise_classes", 0)
    print(f"  hidden={cfg.get('hidden_size')}  depth={cfg.get('depth')}  "
          f"in_channels={in_channels}  size={input_size}")
    print()

    energy_fn  = ENERGIES[args.energy]
    class_ids  = [int(c.strip()) for c in args.classes.split(",") if c.strip()]
    all_rows   = []  # for summary grid: list of (tag, img_base, img_m1, img_m2)

    print("Generating (JIT compile on first call – may take ~30s on CPU)...")
    print()

    for i, class_id in enumerate(class_ids):
        tag   = IMAGENET_LABELS.get(class_id, f"cls{class_id}")
        label = jnp.array([class_id], dtype=jnp.int32)
        rng   = jax.random.PRNGKey(args.seed + i)

        # Split RNG matching DitGen.__call__ convention
        rng_eps, rng_labels = jax.random.split(rng)

        # Fixed epsilon – shared across all methods so comparisons are fair
        epsilon = jax.random.normal(
            rng_eps, (1, input_size, input_size, in_channels)
        )

        # Conditioning (fixed noise_labels so the comparison is clean)
        cond, noise_labels = _get_cond(
            model, params, label, args.cfg_scale, rng_labels,
            noise_coords, noise_classes,
        )

        # ── Baseline ──────────────────────────────────────────────────────
        x_base = _forward(model, params, epsilon, cond)
        jax.block_until_ready(x_base)

        # ── Method 1: output guidance (one step, no model re-eval) ────────
        x_m1 = method1_output(x_base, energy_fn, args.alpha_m1)
        jax.block_until_ready(x_m1)

        # ── Method 2: noise-space guidance (n_steps × fwd+bwd) ───────────
        x_m2, eps_guided = method2_noise(
            model, params, epsilon, cond, energy_fn, args.alpha, args.steps
        )
        jax.block_until_ready(x_m2)

        # ── Logging ───────────────────────────────────────────────────────
        e_base = float(energy_fn(x_base.astype(jnp.float32)))
        e_m1   = float(energy_fn(x_m1.astype(jnp.float32)))
        e_m2   = float(energy_fn(x_m2.astype(jnp.float32)))

        print(f"  [{i+1}/{len(class_ids)}] class {class_id:4d} ({tag})")
        for name, x, e in [("baseline  ", x_base, e_base),
                            ("m1-output ", x_m1,   e_m1),
                            ("m2-noise  ", x_m2,   e_m2)]:
            r, g, b = _channel_stats(x)
            delta = f"  ΔE={e - e_base:+.4f}" if name.strip() != "baseline" else ""
            print(f"    {name}  R={r:.3f} G={g:.3f} B={b:.3f}  E={e:+.4f}{delta}")
        print()

        # ── Save individual images ────────────────────────────────────────
        stem    = f"cls{class_id:03d}_{tag}"
        img_b   = _to_uint8(x_base)
        img_m1  = _to_uint8(x_m1)
        img_m2  = _to_uint8(x_m2)

        Image.fromarray(img_b ).save(output_dir / f"{stem}_baseline.png")
        Image.fromarray(img_m1).save(output_dir / f"{stem}_{args.energy}_m1_output.png")
        Image.fromarray(img_m2).save(output_dir / f"{stem}_{args.energy}_m2_noise.png")

        all_rows.append((tag, img_b, img_m1, img_m2))

    # ── Summary grid: rows = classes, cols = (baseline | m1 | m2) ─────────
    n   = len(all_rows)
    H, W = all_rows[0][1].shape[:2]
    grid = np.zeros((n * H, 3 * W, 3), dtype=np.uint8)
    for row, (_, b, m1, m2) in enumerate(all_rows):
        grid[row*H:(row+1)*H,   0:  W] = b
        grid[row*H:(row+1)*H,   W:2*W] = m1
        grid[row*H:(row+1)*H, 2*W:3*W] = m2

    grid_path = output_dir / f"comparison_{args.energy}_m1a{args.alpha_m1}_m2a{args.alpha}_s{args.steps}.png"
    Image.fromarray(grid).save(grid_path)

    print(f"Saved comparison grid (baseline | m1-output | m2-noise):")
    print(f"  {grid_path}")
    print()
    print(f"Column order: baseline  |  m1-output (direct)  |  m2-noise (through model)")


if __name__ == "__main__":
    main()
