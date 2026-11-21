# rink

rink runs [Inkling 2 Mini](https://thinkingmachines.ai/news/inkling-2-mini/)
locally on Apple silicon.

The current version supports Inkling 2 Mini and Metal. It does not support image input or CUDA.

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
| `/continue` | Continue a reply that reached the token limit. |
| `/reset` | Clear the conversation. |
| `/quit` | Exit. |

Chat uses a 262,144-token context, a 16,384-token reply limit, and effort `0.9` by default.
The reply limit includes reasoning tokens. The context limit covers the full conversation.
The KV cache uses FP16. Conversation history stays in memory until reset or exit.

Use `cargo run --release -- chat --help` to see all options.

Inference supports up a 1 million token context window.
