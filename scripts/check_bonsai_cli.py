#!/usr/bin/env python3
"""End-to-end Bonsai tool checks using the native checkpoint on Metal.

Build first: cargo build --release -p rink-cli
Run: python3 scripts/check_bonsai_cli.py
Commands operate only on synthetic files in a fresh temporary directory.
"""
import argparse
import json
from pathlib import Path
import tempfile
import time

from check_cli import Chat, answers, check, png


def run(args, work):
    binary, model = str(args.binary.resolve()), str(args.model.resolve())
    (work / "notes.txt").write_text("skip\nfirst=violet-6921\nsecond=cedar-2846\nskip\n")
    png(work / "image.png", (230, 15, 20))
    base = [binary, "chat", "--model", model, "--context", "8192"]
    records = {}
    with Chat(base + ["--max-tokens", "16", "--effort", "0", "--tool", "bash,read",
                      "--cwd", str(work)], work, "tools-pauses") as chat:
        page = chat.ask("Use read to open notes.txt with start_line 2 and max_lines 2. Tell me both values from the file.")
        check('"start_line":2' in page and '"max_lines":2' in page, page)
        check("2: first=violet-6921" in page and "3: second=cedar-2846" in page, page)
        check("violet-6921" in answers(page) and "cedar-2846" in answers(page), page)
        follow = chat.ask("Without tools, what was the value of first? Reply with only that value.")
        check("violet-6921" in answers(follow) and "Tool " not in follow, follow)
        check("Conversation cleared." in chat.ask("/reset"), "reset failed")
        result = chat.ask("Use bash to run exactly: printf '%s\\n' bonsai-roundtrip >> created.txt. Then use read to inspect created.txt and tell me its contents. Run the command only once.")
        check((work / "created.txt").read_text() == "bonsai-roundtrip\n", result)
        check("Tool bash:" in result and "Tool read:" in result, result)
        check("bonsai-roundtrip" in answers(result), result)
        check(chat.continuations > 0, "token-limit continuation was not exercised")
        missing = chat.ask("Use read to open missing.txt. If it fails, do not retry; tell me the error.")
        check("Tool read:" in missing and "No such file" in missing, missing)
        image = chat.ask("Use read to open image.png. If image input is unsupported, say so. Do not try another tool.")
        check("does not support image input" in image or "unsupported" in answers(image).lower(), image)
        check("image encoding" not in image.lower(), "Bonsai tried to run image encoding")
        chat.quit()
        records["tools"] = {"checks": ["typed read arguments", "text paging", "cached follow-up",
            "reset", "bash/read roundtrip", "execute once across pauses", "missing file", "unsupported image"],
            "continuations": chat.continuations}
    # Default effort exercises the native thinking -> tool -> thinking transition.
    with Chat(base + ["--tool", "bash", "--cwd", str(work), "--max-tokens", "256"], work, "reasoning-tools") as chat:
        reply = chat.ask("Use bash to run printf 'thinking-tool-ok'. Then reply with exactly its output.")
        check("Tool bash:" in reply and "thinking-tool-ok" in answers(reply), reply)
        check("reasoning: xhigh" in chat.startup, chat.startup)
        check("Conversation cleared." in chat.ask("/reset"), "reset failed")
        cancelled = chat.ask("Use bash to run exactly this command: (sleep 5; printf survived > cancelled-survivor.txt) & wait. Do not change it and do not retry if interrupted. Then tell me the status.",
                             interrupt_on="Tool bash: (sleep 5;")
        check(chat.interrupted and "cancelled true" in cancelled, cancelled)
        check("Tool execution cancelled." in cancelled, cancelled)
        resumed = chat.ask("/continue")
        check("Tool bash: (sleep 5;" not in resumed, "cancelled command was retried")
        chat.quit()
        time.sleep(5.1)
        check(not (work / "cancelled-survivor.txt").exists(), "background child survived")
        records["reasoning"] = ["default xhigh tool round", "SIGINT", "child cleanup", "continue after cancellation"]
    return records


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=root / "target/release/rink")
    parser.add_argument("--model", type=Path, default=root / "checkpoints/bonsai-2.5-27b")
    args = parser.parse_args()
    work = Path(tempfile.mkdtemp(prefix="rink-bonsai-cli-check-"))
    print("Logs and synthetic files:", work, flush=True)
    records = run(args, work)
    (work / "results.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(records, indent=2))
    print("PASS: native Bonsai CLI checks")


if __name__ == "__main__":
    main()
