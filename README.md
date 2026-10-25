# Nanbeige4.3-33B MLX

Streaming conversion and independent validation tools for the unofficial
**Nanbeige4.3-33B MLX affine 4-bit / group64** release.

**[Download the model on Hugging Face](https://huggingface.co/Nurymanau/Nanbeige4.3-33B-MLX-4bit)**
· [Original model](https://huggingface.co/Nanbeige/Nanbeige4.3-33B)

The 14 weight shards occupy **18.63 GB**. This Git repository contains source
code, checksums and validation records. The model weights are hosted on Hugging
Face. The original tokenizer and chat template are preserved.

**Full pretrained-model inference has not been tested by this conversion.**
There are no generation-quality, throughput, full-context, or measured
hardware-fit claims. Quantization can change model outputs and quality.

## Use the published weights

On an Apple Silicon Mac, install the runtime and download the immutable release:

```sh
pip install mlx-lm==0.32.0 mlx==0.32.3 huggingface-hub==1.33.0
hf download Nurymanau/Nanbeige4.3-33B-MLX-4bit \
  --revision af112cf0f59bd372724b26b70f1193fecb9aef6d \
  --local-dir models/nanbeige4.3-33b-mlx
mlx_lm.generate --model models/nanbeige4.3-33b-mlx \
  --prompt "Explain how a hash table works." \
  --max-tokens 512 --temp 0.6 --top-p 0.95
```

Weight files alone exceed 16 GB. Runtime and KV cache require additional memory;
long contexts can require substantially more. Hardware fit has not been measured.
The commands above are usage instructions, not a record of a tested generation.

The upstream chat template retains `enable_thinking`, `preserve_thinking`, and
XML/JSON tool-call formatting. Integration with agent frameworks and tool parsers
has not been tested here.

## What was verified

- Original source pinned to `c1093c501cc6460427c0ac634db7223d0450c0ee`.
- All 23 selected source files and all converted weight files checked by SHA256.
- Every header, offset, dtype, tensor shape and index entry audited:
  471 original tensors / 33,118,977,024 parameters → 1,203 MLX tensors.
- The first two rows of all 366 quantized matrices independently reproduced with
  NumPy: integer codes, BF16 scales and biases match exactly.
- All 105 normalization tensors preserved exactly; six tokenizer/template
  fixtures match between the original and converted tokenizers.
- A synthetic two-layer Llama matches the official `mlx-lm` CPU quantizer on all
  53 tensors and passes strict weight loading. No forward/generation call is made.

See [validation](docs/VALIDATION.json), [tokenizer checks](docs/TOKENIZER_VALIDATION.json)
and [conversion provenance](docs/CONVERSION.json). The
[HF manifest](docs/HF_MANIFEST.json) describes the immutable model artifact on
Hugging Face. `SOURCE_MANIFEST.json` describes this source repository.

## Reproduce the conversion

The converter processes one shard at a time in 64-row CPU chunks; it does not
instantiate the pretrained model. Outputs are written atomically and can be
resumed using their SHA receipts. macOS readers use `F_NOCACHE` for large scans.

Follow [the reproduction guide](docs/REPRODUCING.md). Downloading and converting
the original requires approximately 66.26 GB for source files plus 18.63 GB for
the output, with additional room for temporary downloads. It is optional when
using the already published MLX weights.

## Attribution and license

Original model: **Nanbeige LLM Lab**. This is an unofficial weight conversion of
the standard Llama architecture, with no training or distillation. Published
upstream benchmark results belong to the original authors and were not rerun.

Apache License 2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE).
