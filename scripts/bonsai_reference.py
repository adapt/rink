#!/usr/bin/env python3
"""Generate independent Bonsai references using the published MLX runtime.

Install the checkpoint's runtime/requirements.txt into a temporary environment.
Pass --runtime /path/to/checkpoint/runtime (revision aa432fd0e17612f6684e728d4d3c42b6656bcabb).
This script does not execute or import rink code. Its synthetic fixture exercises
both attention types, grouped recurrent heads, rotation, and a 256-token boundary.
"""
import argparse
import json
from pathlib import Path
import sys

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import numpy as np
from mlx_vlm.models.qwen3_5.language import LanguageModel
from mlx_vlm.models.qwen3_5.config import TextConfig


def install(model, config, weights, packed_class):
    for record in config["modules"]:
        parts = record["path"].split(".")
        parent = model
        for part in parts[:-1]:
            parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
        key = "language_model." + record["path"]
        arrays = [weights[key + "." + suffix] for suffix in ("weight", "scales", "biases")]
        setattr(parent, parts[-1], packed_class(arrays, record["block"], weights.get(key + ".signs"), record["embedding"], mx.float16))
    model.load_weights([(k.removeprefix("language_model."), v.astype(mx.float16) if mx.issubdtype(v.dtype, mx.floating) else v)
                        for k, v in weights.items() if k.startswith("language_model.")], strict=True)
    model.eval()
    mx.eval(model.parameters())


def fixture(source_config, output):
    config = json.loads(source_config.read_text())
    t = config["text_config"]
    t.update(hidden_size=512, intermediate_size=1024, num_hidden_layers=2,
             num_attention_heads=4, num_key_value_heads=1, head_dim=128,
             linear_num_key_heads=1, linear_num_value_heads=3,
             linear_key_head_dim=128, linear_value_head_dim=128,
             vocab_size=256, eos_token_id=0, bos_token_id=0, max_position_embeddings=512, full_attention_interval=2,
             layer_types=["linear_attention", "full_attention"])
    # The fixture is text only; the checkpoint namespace still matches the release.
    config["components"] = {"text": True, "vision": False, "mtp": False}
    config.pop("vision_config", None)
    model = LanguageModel(TextConfig.from_dict(t))
    rng = np.random.default_rng(20261124)
    modules = []
    weights = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Embedding)) and not name.endswith(("in_proj_a", "in_proj_b")):
            rows, width = module.weight.shape
            block = 512 if width % 512 == 0 else 128
            modules.append(dict(path=name, block=block, embedding=isinstance(module, nn.Embedding), dtype="float16"))
            codes = rng.integers(0, 3, (rows, width), dtype=np.uint32)
            packed = np.bitwise_or.reduce(codes.reshape(rows, width//16, 16) << (2*np.arange(16, dtype=np.uint32)), axis=-1)
            scale = rng.uniform(0.01, 0.04, (rows, width//128)).astype(np.float16)
            signs = rng.choice([-1.0, 1.0], width).astype(np.float32)
            key = "language_model." + name
            for suffix, array in [("weight", packed), ("scales", scale), ("biases", -scale), ("signs", signs)]:
                weights[key+"."+suffix] = mx.array(array)
    packed_paths = {r["path"] for r in modules}
    for name, array in tree_flatten(model.parameters()):
        if name.removesuffix(".weight") in packed_paths:
            continue
        if name.endswith("A_log"):
            data = rng.uniform(-1, 1, array.shape)
        elif name.endswith("dt_bias"):
            data = rng.uniform(-2, 0, array.shape)
        elif "norm.weight" in name:
            data = rng.uniform(0.8, 1.2, array.shape)
        else:
            data = rng.normal(0, 0.06, array.shape)
        weights["language_model."+name] = mx.array(data.astype(np.float32))
    config["modules"] = modules
    output.mkdir(parents=True, exist_ok=True)
    (output/"config.json").write_text(json.dumps(config, indent=2)+"\n")
    mx.save_safetensors(str(output/"model.safetensors"), weights)
    return model, config, weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--tokens", type=Path, help="JSON array of token IDs for a native checkpoint comparison")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.runtime.resolve()))
    from runtime import Packed
    if args.fixture:
        model, config, weights = fixture(args.model/"config.json", args.fixture)
    else:
        config = json.loads((args.model/"config.json").read_text())
        model = LanguageModel(TextConfig.from_dict(config["text_config"]))
        weights = mx.load(str(args.model/"model.safetensors"))
    install(model, config, weights, Packed)
    del weights
    tokens = json.loads(args.tokens.read_text()) if args.tokens else [(i*37+11)%256 for i in range(300)]
    cache = model.make_cache()
    points = {}
    for i, token in enumerate(tokens):
        # Token-at-a-time oracle also tests preservation of both recurrent caches.
        hidden = model.model(mx.array([[token]]), cache=cache)
        logits = model.lm_head(hidden)
        mx.eval(logits)
        if i < 16 or i+1 in (127, 255, 256, 257, len(tokens)):
            points[str(i+1)] = np.asarray(logits).astype(np.float32).reshape(-1).tolist()
        if (i+1)%32 == 0:
            print(f"reference {i+1}/{len(tokens)}", flush=True)
    args.output.write_text(json.dumps(dict(tokens=tokens, logits=points,
        reference="prism-ml/Ternary-Bonsai-2.5-27B-mlx-2bit@aa432fd0e17612f6684e728d4d3c42b6656bcabb; mlx 0.32.0; mlx-vlm 0.6.3"))+"\n")


if __name__ == "__main__":
    main()
