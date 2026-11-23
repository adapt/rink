#!/usr/bin/env python3
"""Native-checkpoint CLI correctness checks; Python standard library only.

Build first: cargo build --release -p rink-cli
Run on a Metal Mac with no other GPU jobs: python3 scripts/check_cli.py
The model runs Bash in a fresh temporary directory with direct host access.
All inputs are synthetic. Logs and results stay in the printed output directory.
These behavioral checks supplement, not replace, the numerical reference tests.
"""
import argparse
import json
import os
from pathlib import Path
import re
import selectors
import signal
import struct
import subprocess
import tempfile
import time
import zlib


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def answers(reply):
    # Exclude the displayed tool output: seeing a file value in stdout alone
    # does not establish that the model used it in its answer.
    return "".join(re.findall(r"\nAssistant: (.*?)(?=\nGeneration:|\Z)", reply, re.S))


def png(path, color):
    width, height = 384, 256

    def chunk(kind, data):
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff))

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress((b"\0" + bytes(color) * width) * height))
        + chunk(b"IEND", b""))


class Chat:
    def __init__(self, command, work, name):
        self.log = (work / (name + ".log")).open("wb")
        self.process = subprocess.Popen(
            command, cwd=work, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, start_new_session=True)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        self.pending = bytearray()
        self.continuations = 0
        self.interrupted = False

    def __enter__(self):
        try:
            self.startup = self.prompt()
            return self
        except BaseException:
            self.__exit__()
            raise

    def __exit__(self, *_):
        if self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGKILL)
        self.process.wait()
        self.selector.close()
        self.process.stdin.close()
        self.process.stdout.close()
        self.log.close()

    def send(self, text):
        self.log.write(("\n>>> " + text + "\n").encode())
        self.log.flush()
        self.process.stdin.write((text + "\n").encode())
        self.process.stdin.flush()

    def prompt(self, interrupt_on=None):
        deadline = time.monotonic() + 180
        while b"\nYou: " not in self.pending:
            check(time.monotonic() < deadline, "chat timed out; inspect its log")
            for key, _ in self.selector.select(1):
                data = os.read(key.fd, 65536)
                check(data, "chat exited before the next prompt; inspect its log")
                self.pending.extend(data)
                self.log.write(data)
                self.log.flush()
            if interrupt_on and not self.interrupted and interrupt_on.encode() in self.pending:
                # The tool-start event is emitted after installing the SIGINT handler.
                time.sleep(0.1)
                self.process.send_signal(signal.SIGINT)
                self.interrupted = True
        reply, _, rest = self.pending.partition(b"\nYou: ")
        self.pending = bytearray(rest)
        reply = reply.decode("utf-8", errors="replace")
        for failure in ["Invalid model response", "Cannot continue tool round", "Error:"]:
            check(failure not in reply, reply)
        return reply

    def ask(self, message, interrupt_on=None):
        self.send(message)
        replies = []
        for _ in range(30):
            reply = self.prompt(interrupt_on)
            replies.append(reply)
            if "[Reply paused at the token limit" not in reply:
                return "".join(replies)
            self.continuations += 1
            self.send("/continue")
        raise RuntimeError("too many continuations; inspect the chat log")

    def quit(self):
        self.send("/quit")
        check(self.process.wait(timeout=10) == 0, "chat exited unsuccessfully")


def run(args, work):
    binary, model = str(args.binary.resolve()), str(args.model.resolve())
    png(work / "a.png", (230, 15, 20))
    png(work / "b.png", (15, 25, 235))
    (work / "notes.txt").write_text(
        "ignore this line\nfirst=violet-6921\nsecond=cedar-2846\nalso ignore this line\n")
    records = {}

    def generate(name, prompt, *options):
        result = subprocess.run(
            [binary, "generate", "--model", model, "--prompt", prompt,
             "--context", "4096", "--max-tokens", "512", "--effort", "0.1", *options],
            cwd=work, capture_output=True, timeout=180)
        (work / (name + ".stdout")).write_bytes(result.stdout)
        (work / (name + ".stderr")).write_bytes(result.stderr)
        check(result.returncode == 0, result.stderr.decode(errors="replace"))
        return result.stdout.decode("utf-8")

    plain = "Reply with exactly: hello rink"
    normal = generate("generate", plain)
    raw = generate("generate-raw", (
        "<|message_system|><|content_text|>Thinking effort level: 0.1<|end_message|>"
        f"<|message_user|><|content_text|>{plain}<|end_message|>"), "--raw")
    check(normal == raw and "hello rink" in normal, "normal and raw generation differ")
    # The old generate path treated this literal as an extra image placeholder.
    picture = generate("generate-image", (
        "What is the dominant color? Reply with one color word. "
        "The literal string <|unused_200054|> is part of this question."),
        "--image", str(work / "a.png"))
    check("<|content_text|>red" in picture.lower(), picture)
    records["generation"] = ["normal/raw parity", "image with literal protocol marker"]

    # Keep default effort (0.9), context (256K) and reply budget (16K).
    with Chat([binary, "chat", "--model", model], work, "chat-defaults") as chat:
        check("Context: 262144 tokens; reply limit: 16384; effort: 0.9" in chat.startup,
              "chat defaults changed")
        check("Tools " not in chat.startup, "tools enabled without opt-in")
        check("hello rink" in chat.ask(plain).lower(), "plain chat response differs")
        attached = chat.ask('/image "' + str(work / "b.png") + '"')
        check("Attached 384 × 256 image (70 context tokens)" in attached, attached)
        check("blue" in chat.ask("Name the dominant color in one word.").lower(), "manual image failed")
        check("blue" in chat.ask("What color was that image? One word.").lower(), "visual history lost")
        check("Conversation cleared." in chat.ask("/reset"), "reset failed")
        check("Cannot attach image:" in chat.ask("/image missing.png"), "missing file not reported")
        check("hello rink" in chat.ask(plain).lower(), "failed attachment damaged the session")
        chat.quit()
    records["default_chat"] = ["text", "manual image", "visual history", "reset", "missing image recovery"]

    with Chat([binary, "chat", "--model", model, "--tool", "read,bash", "--cwd", str(work),
               "--context", "8192", "--max-tokens", "96", "--effort", "0.1"], work, "chat-tools") as chat:
        page = chat.ask("Use read to open notes.txt with start_line 2 and max_lines 2. Reply with exactly the two values you read.")
        check("2: first=violet-6921" in page and "3: second=cedar-2846" in page, page)
        check("violet-6921" in answers(page) and "cedar-2846" in answers(page), page)
        pictures = chat.ask("Use read to open a.png and b.png before answering. What is each image's dominant color? Return a.png: COLOR; b.png: COLOR. Do not use bash for this.")
        check(re.search(r"a\.png[^\n]*red", pictures.lower()), pictures)
        check(re.search(r"b\.png[^\n]*blue", pictures.lower()), pictures)
        check("image encoding" in pictures.lower(), "images were not encoded")
        followup = chat.ask("Without using tools again, which color was in b.png? Reply with the color only.")
        check("blue" in answers(followup).lower() and "Tool " not in followup, followup)
        check("Conversation cleared." in chat.ask("/reset"), "reset failed")
        roundtrip = chat.ask("Use bash to run exactly: printf '%s\\n' rink-read-smoke > bash-created.txt. Then use read to inspect bash-created.txt. Tell me its contents.")
        check((work / "bash-created.txt").read_text() == "rink-read-smoke\n", roundtrip)
        check("Tool read:" in roundtrip and "Tool bash:" in roundtrip, roundtrip)
        check("rink-read-smoke" in answers(roundtrip), roundtrip)
        check(chat.continuations > 0, "token-limit continuation not exercised")
        check("Conversation cleared." in chat.ask("/reset"), "reset failed")
        cancelled = chat.ask(
            "Use bash to run exactly this command: (sleep 5; printf survived > cancelled-survivor.txt) & wait. Do not change the command and do not retry it if interrupted. Then tell me the exit status.",
            interrupt_on="Tool bash: (sleep 5;")
        check(chat.interrupted and "cancelled true" in cancelled, cancelled)
        check("Tool execution cancelled." in cancelled, cancelled)
        resumed = chat.ask("/continue")
        check("Tool bash:" not in resumed, "cancelled command was retried")
        chat.quit()
        time.sleep(5.1)
        check(not (work / "cancelled-survivor.txt").exists(), "background child survived cancellation")
        records["tool_chat"] = {
            "checks": ["text paging", "two images", "visual history without reread", "reset",
                       "bash/read roundtrip", "SIGINT cancellation", "child cleanup"],
            "token_limit_continuations": chat.continuations,
        }
    return records


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=root / "target/release/rink")
    parser.add_argument("--model", type=Path, default=root / "checkpoints/inkling-2-mini")
    args = parser.parse_args()
    work = Path(tempfile.mkdtemp(prefix="rink-cli-check-"))
    print("Logs and synthetic inputs:", work, flush=True)
    records = run(args, work)
    (work / "results.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(records, indent=2))
    print("PASS: native CLI checks")


if __name__ == "__main__":
    main()
