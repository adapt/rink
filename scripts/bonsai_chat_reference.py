#!/usr/bin/env python3
"""Generate chat fixtures with the checkpoint's own Transformers/Jinja template.

Usage: python scripts/bonsai_chat_reference.py checkpoints/bonsai-2.5-27b
Requires transformers and tokenizers. Does not load model weights or run the GPU.
"""
import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, default=Path("models/bonsai/tests/fixtures/tools.json"))
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    definitions = [
        {"type": "function", "name": "bash", "description": "Execute a script.",
         "parameters": {"type": "object", "properties": {
             "command": {"type": "string"}, "timeout_ms": {"type": "integer"}},
             "required": ["command"], "additionalProperties": False}},
        {"type": "function", "name": "read", "description": "Read a file.",
         "parameters": {"type": "object", "properties": {
             "path": {"type": "string"}}, "required": ["path"], "additionalProperties": False}}
    ]
    wrapped = [{"type": "function", "function": {k: v for k, v in d.items() if k != "type"}}
               for d in definitions]
    # Rust's serde_json::Value sorts object keys. Preserve those orders when
    # passing declarations to the unmodified released template.
    wrapped = json.loads(json.dumps(wrapped, sort_keys=True))
    calls = [{"function": {"name": "bash", "arguments": {"command": "printf '你好\\n'", "timeout_ms": 1000}}},
             {"function": {"name": "read", "arguments": {"path": "notes.txt"}}}]
    results = ["{\"stdout\":\"你好\\n\",\"exit_code\":0}", "A line from the file."]
    cases = []
    for effort, level in [(0.0, "medium"), (0.1, "low"), (0.5, "medium"), (0.9, "xhigh")]:
        def render(messages, generation=True):
            return tokenizer.apply_chat_template(messages, tools=wrapped, tokenize=False,
                add_generation_prompt=generation, enable_thinking=effort > 0,
                reasoning_effort=level)
        messages = [{"role": "user", "content": " Use both tools. "}]
        first = render(messages)
        messages.append({"role": "assistant", "content": "", "reasoning_content": "",
                         "tool_calls": calls})
        assistant = render(messages, False)
        assert assistant.startswith(first)
        generated = assistant[len(first):].removesuffix("<|im_end|>\n")
        messages.extend({"role": "tool", "content": result} for result in results)
        full = render(messages)
        assert full.startswith(assistant)
        suffix = full[len(assistant) - 1:]  # EOS was cached; its newline was not.
        encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
        cases.append({"effort": effort, "prompt": " Use both tools. ",
                      "tokens": encode(first), "rendered": first,
                      "generated": encode(generated), "suffix": encode(suffix),
                      "generated_rendered": generated, "suffix_rendered": suffix,
                      "results": results})
    data = {"template_sha256": hashlib.sha256((args.model / "chat_template.jinja").read_bytes()).hexdigest(),
            "tools": definitions, "calls": calls, "cases": cases}
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
