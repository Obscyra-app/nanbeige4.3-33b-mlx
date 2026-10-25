# Reproduce and audit the MLX conversion

Reference versions are recorded in [CONVERSION.json](CONVERSION.json).
The scripts target the pinned public source model; they perform no training or
pretrained-model inference. Run each processing step sequentially.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r conversion/requirements.txt
```

Download the source and preserve its upstream SHA256 receipts:

```sh
python conversion/prepare.py \
  --source work/source --receipt work/source.json --xet --workers 3
```

If already logged in with `hf auth login`, append `--authenticated` to use that
login. The script does not copy or print the token. Avoid overriding `HF_HOME`
if you want to retain access to your existing login; use `HF_HUB_CACHE` and
`HF_XET_CACHE` to choose cache locations.

Check the converter on a small synthetic model, without inference:

```sh
python conversion/check_converter.py
```

Convert the verified source in CPU chunks:

```sh
python conversion/convert.py \
  --source work/source --output work/mlx-4bit \
  --receipt work/conversion.json --download-receipts work/source.json
```

Rerun the same command to resume an interrupted conversion. It rechecks prior
source/output SHA values and preserves complete shards. Do not change the source
revision, quantization parameters or files during a run.

Independently audit the complete source and converted package with NumPy:

```sh
python conversion/audit_package.py \
  --source work/source --model work/mlx-4bit \
  --conversion-receipt work/conversion.json \
  --download-receipt work/source.json --output work/validation.json
python conversion/check_tokenizer.py \
  --source work/source --converted work/mlx-4bit \
  --output work/tokenizer-validation.json
```

The audit covers all file identities and tensor layouts. Independent numerical
reproduction samples the first two rows per matrix; it does not evaluate every
row or full-model generation behavior. `check_tokenizer.py` checks six template
scenarios and does not load model weights.

For acquisition split across multiple download receipts, `finalize_source.py`
can merge them and rehash the complete source before the final audit:

```sh
python conversion/finalize_source.py --source work/source \
  --receipts work/source-part1.json work/source-part2.json \
  --output work/source-complete.json
```

The converter produces standard `mlx-lm` weights, configuration and tokenizer
files. Model-card publication and the runtime generation example are separate
from these conversion checks.
