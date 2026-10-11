# Security

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub:
[Report a vulnerability](https://github.com/TKY-27/iris-3b-metal/security/advisories/new).
Do not open a public issue. Include the affected version or commit, how to
reproduce the problem and what an attacker gains. Reports are handled on a
best-effort basis.

Only the latest release and `main` receive fixes.

## What is in scope

- `iris-metal serve`, the local HTTP API. It has no authentication and is
  meant for `127.0.0.1` only. It refuses requests that carry an `Origin`
  header or a `Sec-Fetch-Site` header other than `none`, and requests for host
  names other than localhost and the address it is bound to, so that web pages
  and DNS rebinding cannot reach it. A way around those checks, or a request
  that crashes the server or reads or writes files it should not, is a
  vulnerability. Exposing the server on another interface with `--host` is
  unsupported.
- The macOS app's handling of its library (`~/Library/Application Support/Iris`)
  and of the images users add to it.
- Model loading. The engine reads `safetensors` files only, from the
  Hugging Face revisions pinned in `src/iris_metal/cli.py` and
  `src/iris_metal/text.py` unless the user chooses others.

## Known limitations

- The upstream training, resume, sampling and export scripts under
  `scripts/` and `src/iris3b/` load `.pth` files with pickle. Only load
  checkpoints you trust.
- Weights downloaded from Hugging Face are trusted as published by
  [speridlabs/iris-3b](https://huggingface.co/speridlabs/iris-3b) and
  [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct).
- The macOS app is not distributed as a signed or notarized binary; build it
  from source.
