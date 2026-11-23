# rink

rink runs [Inkling 2 Mini](https://thinkingmachines.ai/news/inkling-2-mini/)
locally on Apple silicon.

The current version supports text and image input with Inkling 2 Mini on Metal.
Support for CUDA and other models is planned.

## Requirements

- macOS on Apple silicon.
- A Rust toolchain with support for edition 2024.
- The Hugging Face CLI (`hf`) to download the model files.
- About 20 GB of disk space for the model files.

Use a Mac with 48 GB or more unified memory for the default chat settings.
For a 32 GB Mac, use the smaller context below.

## Run

Download the original model files:

```sh
hf download thinkingmachines/Inkling-2-Mini \
  --revision 6e7c8248f75265bdd4fbab5273304243b5d8c2cb \
  --local-dir checkpoints/inkling-2-mini \
  --include config.json tokenizer.json model.safetensors.index.json model-00001-of-00001.safetensors
```

Start a chat:

```sh
cargo run --release -- chat
```

For a 32 GB Mac, start with a smaller context and reply limit:

```sh
cargo run --release -- chat --context 4096 --max-tokens 1024
```

## Chat

Enter one message per line. Replies appear as the model generates them.

| Command | Action |
| --- | --- |
| `/help` | Show the commands. |
| `/image PATH` | Attach a PNG or JPEG to the next message. |
| `/image clear` | Remove the pending image. |
| `/continue` | Continue a reply that reached the token limit. |
| `/reset` | Clear the conversation. |
| `/quit` | Exit. |

Chat uses a 262,144-token context, a 16,384-token reply limit, and effort `0.9` by default.
The reply limit includes reasoning tokens. The context limit covers the full conversation.
The KV cache uses FP16. Conversation history stays in memory until reset or exit.
Prefill uses batches of up to 1,024 tokens. This targets about 30 GiB of Metal
memory at 256K context. Use `--prefill-batch 128` to reduce memory use. See the
[performance measurements](benchmarks/prefill-30g.md).

To ask about a picture, enter `/image /path/to/picture.png`, then enter your question.
Each message can have one image. Images must use 8-bit channels and can be up to 16,384 pixels on each side.
rink keeps the original image size. Large images use more memory and take longer to process.
Chat shows progress and separate times for image encoder loading, image encoding,
prompt prefill, and generation. It also shows when the model is thinking.
Progress goes to stderr. Use `--show-thinking` to read the reasoning text.

Use `cargo run --release -- chat --help` to see all options.

The model configuration allows up to 1,048,576 tokens. This context does not fit
in 48 GB of unified memory.

## Tools

Enable file reading and Bash:

```sh
cargo run --release -- chat --tool read,bash --cwd /path/to/project
```

Ask the model to read a file by path. `read` returns UTF-8 text with line numbers,
or adds a PNG or JPEG directly to the model's image context. Relative paths use
`--cwd`; `~/` uses your home directory. Use `--tool read` to enable file reading
alone. PDFs and other binary formats are not supported.

For text, the model can select `start_line` (starting at 1) and `max_lines` (200 by
default, up to 2,000). Long results include a continuation line or a truncation
message. `--tool-output-bytes` sets the text byte limit. Images use the same
pixel limits and processing as `/image`, with a 2 GiB limit on the encoded file.
Images must fit as complete inputs in the remaining context. They are processed
one at a time. Temporary image snapshots are removed after use.

Enable Bash so the model can run commands:

```sh
cargo run --release -- chat --tool bash --cwd /path/to/project
```

Commands run directly with your user permissions. There is no approval prompt or
sandbox. Each call starts a new shell in `--cwd` (the current directory by default).
Shell variables and directory changes do not carry over to the next call.
Commands cannot read terminal input. Background processes in the command's process
group stop when the call ends. Chat shows each command and its output.

The defaults are a 120-second tool timeout, 16 KiB of text per call (Bash splits
this between stdout and stderr), and eight tool rounds per user message. Set these with
`--tool-timeout-ms`, `--tool-output-bytes`, and `--max-tool-rounds`.
Press Ctrl-C during a command to cancel it. Use `/continue` to let the model read
the result. `/continue` also resumes a reply that reached its token limit.
Tool results use context space. The reply limit includes thinking and tool calls.

## AI disclosure

The vast majority of this code was written by GPT 6 Astra with human guidance.
