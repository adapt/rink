#!/usr/bin/env python3
"""Independent CPU PyTorch reference for Inkling's original vision weights.

Equations and preprocessing follow Transformers PR #51050, commit
6173a2f679ff4ed393afc499aefc569da7e68bc1. Keep original BF16 weight values,
but use F32 arithmetic throughout, like rink. No approximate/tanh GELU.

Requires torch, numpy, Pillow, safetensors. Two uses:
  python scripts/vision_reference.py fixture
  python scripts/vision_reference.py native CHECKPOINT_DIR OUTPUT_DIR
The latter writes full-encoder references for compare_vision (no model download).
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from safetensors import safe_open
from safetensors.torch import load_file, save_file

torch.set_num_threads(4)
torch.set_grad_enabled(False)
ROOT = Path(__file__).resolve().parents[1]


def pixels(image):
    rgba = np.array(ImageOps.exif_transpose(image).convert("RGBA"), dtype=np.int32)
    height, width = rgba.shape[:2]
    yy, xx = np.indices((height, width))
    background = np.where((yy // 8 + xx // 8) % 2 == 0, 204, 153)[..., None]
    rgb = (rgba[..., :3] * rgba[..., 3:] + background * (255 - rgba[..., 3:]) + 127) // 255
    rows, cols = math.ceil(height / 40), math.ceil(width / 40)
    padded = torch.full((rows * 40, cols * 40, 3), -1., dtype=torch.float32)
    padded[:height, :width] = torch.from_numpy(rgb.astype(np.float32))
    padded = (padded - 127.5) / 127.5
    patches = padded.reshape(rows, 2, 20, cols, 2, 20, 3).permute(0, 3, 1, 4, 2, 5, 6)
    return patches.reshape(-1, 1200), rows * 2, cols * 2


def encode(image, config, weights):
    def w(name):
        return weights["model.visual." + name].float()

    def linear(x, name, bias=True):
        return F.linear(x, w(name + ".weight"), w(name + ".bias") if bias else None)

    def norm(x, name):
        return F.layer_norm(x, (config["hidden_size"],), w(name + ".weight"),
                            w(name + ".bias"), config["layer_norm_eps"])

    patches, rows, cols = pixels(image)
    h, heads = config["hidden_size"], config["num_attention_heads"]
    dim, side = h // heads, config["posemb_grid_size"]
    # Raster coordinates, then regroup into the processor's 2x2 tile order.
    yy, xx = torch.meshgrid(torch.arange(rows), torch.arange(cols), indexing="ij")
    def ordered(v):
        return v.reshape(rows // 2, 2, cols // 2, 2).permute(0, 2, 1, 3).flatten()
    yy, xx = ordered(yy), ordered(xx)
    fy, fx = (yy.float() + .5) * side / rows - .5, (xx.float() + .5) * side / cols - .5
    y0, x0 = fy.floor().long(), fx.floor().long()
    dy, dx = fy - y0, fx - x0
    indices, factors = [], []
    for a in range(2):
        for b in range(2):
            indices.append((y0 + a).clamp(0, side - 1) * side + (x0 + b).clamp(0, side - 1))
            factors.append((dy if a else 1-dy) * (dx if b else 1-dx))
    positions = (w("backbone.embeddings.position_embedding")[torch.stack(indices, 1)] *
                 torch.stack(factors, 1)[..., None]).sum(1)
    x = linear(patches, "backbone.embeddings.patch_embedding", False) + positions
    cls = w("backbone.embeddings.class_embedding") + w("backbone.embeddings.cls_position_embedding")
    x = norm(torch.cat((cls[None], x)), "backbone.ln_pre")
    inverse = 1 / (10000 ** (torch.arange(0, dim // 2, 2).float() / (dim // 2)))
    xy = torch.stack((xx + 1, yy + 1), 1)
    xy = torch.cat((torch.zeros((1, 2)), xy))
    angles = (xy[..., None] * inverse).reshape(-1, dim // 2).repeat_interleave(2, -1)
    cos, sin = angles.cos()[:, None], angles.sin()[:, None]
    def rotate(q):
        pairs = q.reshape(q.shape[0], heads, -1, 2)
        rotated = torch.stack((-pairs[..., 1], pairs[..., 0]), -1).flatten(-2)
        return q * cos + rotated * sin
    for layer in range(config["num_hidden_layers"]):
        p = f"backbone.layers.{layer}"
        n = norm(x, p + ".ln_1")
        q, k, v = [linear(n, p + ".self_attn." + kind + "_proj").reshape(-1, heads, dim)
                   for kind in ("q", "k", "v")]
        q, k, v = rotate(q).transpose(0, 1), rotate(k).transpose(0, 1), v.transpose(0, 1)
        # Deliberately materialize the full noncausal score matrix in the oracle.
        probabilities = ((q @ k.transpose(-1, -2)) * (dim ** -.5)).softmax(-1)
        attn = (probabilities @ v).transpose(0, 1).reshape(-1, h)
        x = x + w(p + ".ls_1.gamma") * linear(attn, p + ".self_attn.out_proj")
        up = F.gelu(linear(norm(x, p + ".ln_2"), p + ".mlp.c_fc"), approximate="none")
        x = x + w(p + ".ls_2.gamma") * linear(up, p + ".mlp.c_proj")
    features = linear(F.gelu(linear(x[1:].reshape(-1, 4*h), "proj_in", False), approximate="none"), "proj_out", False)
    return {"patches": patches.contiguous(), "features": features.contiguous()}


def pattern(width, height, alpha=True):
    y, x = np.indices((height, width))
    a = (x * 7 + y * 17) % 256 if alpha else np.full_like(x, 255)
    rgba = np.stack(((x * 11 + y) % 256, (y * 13 + x) % 256, (x * 3 + y * 5) % 256, a), -1)
    return Image.fromarray(rgba.astype(np.uint8))


def fixture():
    import make_test_fixture as text
    out = ROOT / "models/inkling/tests/fixtures/vision"
    out.mkdir(exist_ok=True)
    c = dict(model_type="inkling2_vision", native_patch_size=20, adapter_merge_size=2,
             hidden_size=32, num_attention_heads=2, num_hidden_layers=2, mlp_ratio=2.,
             posemb_grid_size=3, layer_norm_eps=1e-5, projection_hidden_dim=48,
             projector_bias=False, text_hidden_size=8)
    generator = torch.Generator().manual_seed(987654)
    weights = {}
    def put(name, shape, scale=.1, center=0.):
        weights["model.visual." + name] = (torch.randn(shape, generator=generator) * scale + center).bfloat16()
    put("backbone.embeddings.patch_embedding.weight", (32, 1200), .03)
    put("backbone.embeddings.position_embedding", (9, 32))
    for name in ("class_embedding", "cls_position_embedding"):
        put("backbone.embeddings." + name, (32,))
    put("backbone.ln_pre.weight", (32,), .1, 1.)
    put("backbone.ln_pre.bias", (32,))
    for i in range(2):
        p = f"backbone.layers.{i}"
        for ln in ("ln_1", "ln_2"):
            put(f"{p}.{ln}.weight", (32,), .1, 1.)
            put(f"{p}.{ln}.bias", (32,))
        for ls in ("ls_1", "ls_2"):
            put(f"{p}.{ls}.gamma", (32,), .05, .1)
        for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
            put(f"{p}.self_attn.{proj}.weight", (32, 32))
            put(f"{p}.self_attn.{proj}.bias", (32,))
        for proj, rows, cols in (("c_fc", 64, 32), ("c_proj", 32, 64)):
            put(f"{p}.mlp.{proj}.weight", (rows, cols))
            put(f"{p}.mlp.{proj}.bias", (rows,))
    put("proj_in.weight", (48, 128))
    put("proj_out.weight", (8, 48))
    image = pattern(241, 81)
    image.save(out / "input.png")
    orientation = Image.Exif()
    orientation[274] = 6
    pattern(11, 7, False).convert("RGB").save(out / "oriented.jpg", quality=95, subsampling=0, exif=orientation)
    ImageOps.exif_transpose(Image.open(out / "oriented.jpg")).save(out / "oriented-expected.png")
    Image.fromarray(np.array([[1024]], dtype=np.uint16)).save(out / "unsupported-16bit.png")
    expected = encode(image, c, weights)
    save_file(expected, out / "expected.safetensors")
    config = dict(text.config, image_token_id=13, vision_config=c)
    config["text_config"]["model_max_length"] = 128
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    weights.update(load_file(text.OUT / "model.safetensors"))
    save_file(weights, out / "model.safetensors")
    # Image begins at offset 17 and crosses a 32-token prefill chunk boundary.
    # A second image comes in a later turn. Compare final prefill logits and
    # every continuation against the independent scalar decoder.
    tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5] + [13] * 21 + [2, 3, 4]
    turn2 = [2] + [13] * 21 + [4]
    for fp16 in (False, True):
        text.fp16_cache = fp16
        text.histories.clear()
        text.kv_cache.clear()
        pos = 0
        def run(ids, start=None):
            nonlocal pos
            results = []
            for i, token in enumerate(ids):
                embedding = expected["features"][i-start].tolist() if start is not None and start <= i < start+21 else None
                results.append(text.step(token, pos, embedding))
                pos += 1
            return results
        first = run(tokens, 17)[-1]
        continuation = run([4, 5, 6])
        second = run(turn2, 1)[-1]
        after = run([7, 8])
        (out / ("text-f16.json" if fp16 else "text-f32.json")).write_text(json.dumps(dict(
            tokens=tokens, prefill=first, continuation=continuation, turn2=turn2,
            second=second, after=after), indent=2) + "\n")
    print("Wrote vision and mixed image/text fixtures to", out)


def native(directory, output):
    directory, output = Path(directory), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    c = json.loads((directory / "config.json").read_text())["vision_config"]
    index = json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"]
    weights = {}
    for shard in sorted({v for k, v in index.items() if k.startswith("model.visual.")}):
        with safe_open(directory / shard, framework="pt") as f:
            weights.update({k: f.get_tensor(k) for k in f.keys() if k.startswith("model.visual.")})
    for width, height, alpha in [(83, 47, False), (41, 79, True), (641, 41, True), (400, 320, False)]:
        name = f"{width}x{height}"
        image = pattern(width, height, alpha)
        image.save(output / (name + ".png"))
        save_file(encode(image, c, weights), output / (name + ".safetensors"))
        print("Wrote", name, flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["fixture"]:
        fixture()
    elif len(sys.argv) == 4 and sys.argv[1] == "native":
        native(*sys.argv[2:])
    else:
        sys.exit(__doc__)
