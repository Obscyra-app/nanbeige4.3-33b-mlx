"""Shard-at-a-time conversion into standard mlx-lm affine weights.

No model is instantiated and no inference is performed. Original shards remain
untouched. Each output is committed atomically, with a resumable SHA receipt.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import struct
import time

import mlx.core as mx
import numpy as np

from prepare import REVISION, REPO, atomic_json, digest, uncached_reader


def convert_shard(source, destination, bits=4, group_size=64):
    mx.set_default_device(mx.cpu)
    mx.set_cache_limit(32 * 1024 * 1024)
    with source.open("rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        if not 0 < size < 10_000_000:
            raise ValueError("Invalid safetensors header")
        tensors = json.loads(f.read(size))
    tensors.pop("__metadata__", None)
    source_offset = 8 + size
    specs = {}
    converted = []
    untouched = []
    for name, tensor in sorted(tensors.items()):
        shape = tensor["shape"]
        if tensor["dtype"] != "BF16" or not name.endswith(".weight"):
            raise ValueError(f"Unexpected source tensor: {name} {tensor}")
        if len(shape) == 2:
            height, width = shape
            if width % group_size or (width * bits) % 32:
                raise ValueError(f"Unexpected matrix shape: {name} {shape}")
            prefix = name.removesuffix(".weight")
            specs[name] = {"dtype": "U32", "shape": [height, width * bits // 32]}
            specs[prefix + ".scales"] = {"dtype": "BF16", "shape": [height, width // group_size]}
            specs[prefix + ".biases"] = {"dtype": "BF16", "shape": [height, width // group_size]}
            converted.append({"name": name, "shape": shape, "source_dtype": "mlx.core.bfloat16"})
        elif len(shape) == 1:
            specs[name] = {"dtype": "BF16", "shape": shape}
            untouched.append(name)
        else:
            raise ValueError(f"Unexpected source tensor: {name} {shape}")
    end = 0
    for name in sorted(specs):
        spec = specs[name]
        length = int(np.prod(spec["shape"])) * (4 if spec["dtype"] == "U32" else 2)
        spec["data_offsets"] = [end, end + length]
        end += length
    encoded = json.dumps({"__metadata__": {"format": "mlx"}, **dict(sorted(specs.items()))}, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    target_offset = 8 + len(encoded)
    tmp = destination.with_name(destination.stem + ".partial.safetensors")
    with source.open("rb") as inp, tmp.open("w+b") as out:
        uncached_reader(inp)
        out.write(struct.pack("<Q", len(encoded)))
        out.write(encoded)
        out.truncate(target_offset + end)
        for name, tensor in sorted(tensors.items()):
            shape = tensor["shape"]
            if len(shape) == 1:
                inp.seek(source_offset + tensor["data_offsets"][0])
                length = tensor["data_offsets"][1] - tensor["data_offsets"][0]
                raw = inp.read(length)
                if len(raw) != length:
                    raise ValueError("Truncated source tensor")
                out.seek(target_offset + specs[name]["data_offsets"][0])
                out.write(raw)
                continue
            height, width = shape
            prefix = name.removesuffix(".weight")
            for row in range(0, height, 64):
                count = min(64, height - row)
                inp.seek(source_offset + tensor["data_offsets"][0] + row * width * 2)
                raw = inp.read(count * width * 2)
                if len(raw) != count * width * 2:
                    raise ValueError("Truncated source matrix")
                weight = mx.array(np.frombuffer(raw, dtype="<u2").reshape(count, width)).view(mx.bfloat16)
                packed, scales, biases = mx.quantize(weight, group_size=group_size, bits=bits, mode="affine")
                mx.eval(packed, scales, biases)
                for key, array in [(name, packed), (prefix + ".scales", scales), (prefix + ".biases", biases)]:
                    value = np.asarray(array if specs[key]["dtype"] == "U32" else array.view(mx.uint16))
                    data = value.astype("<u4" if specs[key]["dtype"] == "U32" else "<u2", copy=False).tobytes()
                    row_bytes = specs[key]["shape"][-1] * (4 if specs[key]["dtype"] == "U32" else 2)
                    out.seek(target_offset + specs[key]["data_offsets"][0] + row * row_bytes)
                    out.write(data)
                    del value, data
                del weight, packed, scales, biases, array, raw
            mx.clear_cache()
        out.flush()
        os.fsync(out.fileno())
    result = {"source_file": source.name, "source_sha256": digest(source),
              "file": destination.name, "sha256": digest(tmp), "bytes": tmp.stat().st_size,
              "tensor_bytes": end,
              "converter_sha256": digest(Path(__file__)), "method": "row_chunked_64_cpu",
              "weight_keys": sorted(specs), "quantized": converted, "unchanged": untouched}
    del specs, tensors
    gc.collect()
    mx.clear_cache()
    tmp.replace(destination)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--receipt", required=True, type=Path)
    p.add_argument("--download-receipts", required=True, type=Path, nargs="+")
    p.add_argument("--bits", type=int, choices=[4, 6, 8], default=4)
    p.add_argument("--group-size", type=int, default=64)
    p.add_argument("--follow", action="store_true", help="Wait for verified source shards in this foreground job")
    p.add_argument("--wait-timeout", type=float, default=14400)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    a.receipt.parent.mkdir(parents=True, exist_ok=True)
    if a.receipt.exists():
        receipt = json.loads(a.receipt.read_text())
        if (receipt["revision"], receipt["bits"], receipt["group_size"]) != (REVISION, a.bits, a.group_size):
            raise ValueError("Resume configuration differs")
    else:
        if list(a.output.glob("*.safetensors")):
            raise ValueError("Refusing to reuse an output without its conversion receipt")
        receipt = {"source": REPO, "revision": REVISION, "bits": a.bits,
                   "group_size": a.group_size, "mode": "affine", "shards": [], "complete": False}
    started = time.monotonic()
    while not (a.source / "model.safetensors.index.json").exists():
        if not a.follow or time.monotonic() - started > a.wait_timeout:
            raise RuntimeError("Missing source index")
        time.sleep(5)
    source_index = json.loads((a.source / "model.safetensors.index.json").read_text())
    names = sorted(set(source_index["weight_map"].values()))
    done = {r["source_file"]: r for r in receipt["shards"]}
    for name, result in done.items():
        if digest(a.output / result["file"]) != result["sha256"] or digest(a.source / name) != result["source_sha256"]:
            raise ValueError(f"Resume integrity failure: {name}")
    while len(done) < len(names):
        verified = {}
        for path in a.download_receipts:
            if path.exists():
                proof = json.loads(path.read_text())
                if proof["revision"] != REVISION:
                    raise ValueError("Source verification revision mismatch")
                verified.update({x["name"]: x for x in proof["files"]})
        progress = False
        for name in names:
            if name in done or name not in verified or not (a.source / name).exists():
                continue
            # prepare.py atomically renames a source only after its upstream SHA passes.
            shard_started = time.monotonic()
            result = convert_shard(a.source / name, a.output / name, a.bits, a.group_size)
            if result["source_sha256"] != verified[name]["sha256"]:
                raise ValueError(f"Source changed after download verification: {name}")
            result["elapsed_seconds"] = time.monotonic() - shard_started
            receipt["shards"].append(result)
            done[name] = result
            atomic_json(a.receipt, receipt)
            print(json.dumps({"converted": name, "bytes": result["bytes"], "seconds": result["elapsed_seconds"], "shards_done": len(done), "shards_total": len(names)}), flush=True)
            progress = True
        if len(done) == len(names):
            break
        if not a.follow or time.monotonic() - started > a.wait_timeout:
            raise RuntimeError("Source download incomplete; completed conversions preserved")
        if not progress:
            time.sleep(5)
    actual_source_keys = {x["name"] for r in receipt["shards"] for x in r["quantized"]} | {x for r in receipt["shards"] for x in r["unchanged"]}
    if actual_source_keys != set(source_index["weight_map"]):
        raise ValueError("Source tensor coverage differs from index")
    for name in ["tokenizer.json", "tokenizer.model", "tokenizer_config.json", "added_tokens.json", "special_tokens_map.json", "generation_config.json"]:
        if not (a.source / name).exists():
            raise RuntimeError(f"Missing metadata: {name}")
        shutil.copyfile(a.source / name, a.output / name)
    config = json.loads((a.source / "config.json").read_text())
    q = {"group_size": a.group_size, "bits": a.bits, "mode": "affine"}
    config.update(quantization=q, quantization_config=q)
    atomic_json(a.output / "config.json", config)
    weight_map = {key: r["file"] for r in receipt["shards"] for key in r["weight_keys"]}
    atomic_json(a.output / "model.safetensors.index.json", {"metadata": {"total_size": sum(r["tensor_bytes"] for r in receipt["shards"])}, "weight_map": dict(sorted(weight_map.items()))})
    receipt.update(complete=True, elapsed_seconds=time.monotonic() - started,
                   tensor_count=len(weight_map), weight_bytes=sum(r["bytes"] for r in receipt["shards"]))
    atomic_json(a.receipt, receipt)
    print(json.dumps({"complete": True, "weight_bytes": receipt["weight_bytes"], "tensor_count": receipt["tensor_count"]}), flush=True)


if __name__ == "__main__":
    main()
