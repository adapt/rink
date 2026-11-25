#!/usr/bin/env python3
"""Benchmark the published Bonsai MLX components on the same text as `rink bench`.

Uses the model pack's runtime/requirements.txt. The cache grows through MLX's own
cache implementation; rink reserves its cache at load time. No MTP is enabled.
"""
import argparse
import json
from importlib.metadata import version
from pathlib import Path
import sys
import time

import mlx.core as mx
from tokenizers import Tokenizer

from bonsai_reference import LanguageModel, TextConfig, install


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--prompt-tokens", default="128,512,1024")
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    lengths = [int(value) for value in args.prompt_tokens.split(",")]
    if min(lengths) < 1 or args.decode_tokens < 1 or args.repetitions < 1:
        parser.error("prompt lengths, decode length and repetitions must be positive")
    sys.path.insert(0, str(args.runtime.resolve()))
    from runtime import Packed, install_shared_rotations

    config = json.loads((args.model / "config.json").read_text())
    model = LanguageModel(TextConfig.from_dict(config["text_config"]))
    weights = mx.load(str(args.model / "model.safetensors"))
    install(model, config, weights, Packed)
    del weights
    sharing = install_shared_rotations(model)
    tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    ids = tokenizer.encode(
        "Explain how a computer processes information, using clear examples. ",
        add_special_tokens=False,
    ).ids
    cache = model.make_cache()
    for token in ids[:4]:
        hidden = model.model(mx.array([[token]]), cache=cache)
        mx.eval(model.lm_head(hidden))
    del cache

    trials = []
    for length in lengths:
        prompt = (ids * ((length + len(ids) - 1) // len(ids)))[:length]
        for repetition in range(args.repetitions):
            cache = model.make_cache()
            start = time.perf_counter()
            hidden = model.model(mx.array([prompt]), cache=cache)
            logits = model.lm_head(hidden[:, -1:, :])
            next_token = mx.argmax(logits[:, -1, :], axis=-1)
            current = next_token.item()
            prefill_seconds = time.perf_counter() - start
            generated = []
            start = time.perf_counter()
            for _ in range(args.decode_tokens):
                generated.append(current)
                hidden = model.model(next_token[:, None], cache=cache)
                logits = model.lm_head(hidden)
                next_token = mx.argmax(logits[:, -1, :], axis=-1)
                current = next_token.item()
            decode_seconds = time.perf_counter() - start
            trial = dict(
                prompt_tokens=length, repetition=repetition,
                prefill_seconds=prefill_seconds, decode_seconds=decode_seconds,
                prefill_tokens_per_second=length / prefill_seconds,
                decode_tokens_per_second=args.decode_tokens / decode_seconds,
                generated_ids=generated, peak_memory_bytes=mx.get_peak_memory(),
            )
            trials.append(trial)
            print(json.dumps(trial), file=sys.stderr, flush=True)
    print(json.dumps(dict(
        runtime_versions={name: version(name) for name in ("mlx", "mlx-lm", "mlx-vlm")},
        shared_rotations=sharing, warmup_tokens=4,
        decode_tokens=args.decode_tokens, trials=trials,
    ), indent=2))


if __name__ == "__main__":
    main()
