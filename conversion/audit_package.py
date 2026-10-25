"""Independent NumPy/header audit. Does not import MLX or run the model."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

import numpy as np

DTYPES = {"BF16": 2, "F16": 2, "F32": 4, "U32": 4}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        if sys.platform == "darwin":
            import fcntl
            fcntl.fcntl(f.fileno(), fcntl.F_NOCACHE, 1)
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def header(path):
    with path.open("rb") as f:
        length = struct.unpack("<Q", f.read(8))[0]
        assert 0 < length < 10_000_000, path.name
        data = json.loads(f.read(length))
    data.pop("__metadata__", None)
    end = 0
    for key, spec in sorted(data.items(), key=lambda x: x[1]["data_offsets"][0]):
        start, stop = spec["data_offsets"]
        assert start == end, (path.name, key, "noncontiguous offsets")
        assert stop - start == math.prod(spec["shape"]) * DTYPES[spec["dtype"]], key
        end = stop
    assert 8 + length + end == path.stat().st_size, path.name
    return data, 8 + length


def rows(path, spec, offset, count=2):
    shape = spec["shape"]
    width = shape[-1]
    with path.open("rb") as f:
        f.seek(offset + spec["data_offsets"][0])
        raw = f.read(min(count, shape[0]) * width * DTYPES[spec["dtype"]])
    dtype = {"BF16": "<u2", "F16": "<f2", "F32": "<f4", "U32": "<u4"}[spec["dtype"]]
    data = np.frombuffer(raw, dtype=dtype).reshape(-1, width)
    if spec["dtype"] == "BF16":
        return (data.astype(np.uint32) << 16).view(np.float32)
    return data


def bf16_round(value):
    bits = np.ascontiguousarray(value, dtype=np.float32).view(np.uint32)
    rounded = ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) >> 16).astype(np.uint32)
    return (rounded << 16).view(np.float32)


def independent_affine4(reference):
    # Float32 group math, nearest-even integer rounding and endpoint alignment.
    # Reference specification: mlx v0.32.3 CPU quantize() in
    # https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/cpu/quantized.cpp
    blocks = reference.reshape(-1, 64).astype(np.float32)
    low, high = blocks.min(axis=1), blocks.max(axis=1)
    choose_low = np.abs(low) > np.abs(high)
    step = np.maximum((high - low) / np.float32(15), np.float32(1e-7))
    step = np.where(choose_low, step, -step)
    endpoint = np.where(choose_low, low, high)
    offset_bin = np.rint(endpoint / step)
    adjusted = np.divide(endpoint, offset_bin, out=step.copy(), where=offset_bin != 0)
    bias = np.where(offset_bin != 0, endpoint, np.float32(0))
    integers = np.clip(np.rint((blocks - bias[:, None]) / adjusted[:, None]), 0, 15).astype(np.uint32)
    return integers.reshape(reference.shape), bf16_round(adjusted), bf16_round(bias)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--conversion-receipt", type=Path, required=True)
    p.add_argument("--download-receipt", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    conv = json.loads(a.conversion_receipt.read_text())
    download = json.loads(a.download_receipt.read_text())
    assert conv["complete"] and download["complete"]
    assert conv["revision"] == download["revision"]
    assert conv["bits"] == 4 and conv["group_size"] == 64
    config = json.loads((a.model / "config.json").read_text())
    original_config = json.loads((a.source / "config.json").read_text())
    for key, value in original_config.items():
        assert config[key] == value, key
    assert config["model_type"] == "llama"
    assert config["architectures"] == ["LlamaForCausalLM"]
    assert config["hidden_act"] == "silu" and config["tie_word_embeddings"] is False
    assert not config["attention_bias"] and not config["mlp_bias"]
    hidden, intermediate, vocab = config["hidden_size"], config["intermediate_size"], config["vocab_size"]
    heads, kv_heads, head_dim = config["num_attention_heads"], config["num_key_value_heads"], config["head_dim"]
    expected_shapes = {"model.embed_tokens.weight": [vocab, hidden], "model.norm.weight": [hidden], "lm_head.weight": [vocab, hidden]}
    for i in range(config["num_hidden_layers"]):
        prefix = f"model.layers.{i}."
        for name, shape in {
            "self_attn.q_proj.weight": [heads * head_dim, hidden],
            "self_attn.k_proj.weight": [kv_heads * head_dim, hidden],
            "self_attn.v_proj.weight": [kv_heads * head_dim, hidden],
            "self_attn.o_proj.weight": [hidden, heads * head_dim],
            "mlp.gate_proj.weight": [intermediate, hidden],
            "mlp.up_proj.weight": [intermediate, hidden],
            "mlp.down_proj.weight": [hidden, intermediate],
            "input_layernorm.weight": [hidden], "post_attention_layernorm.weight": [hidden],
        }.items():
            expected_shapes[prefix + name] = shape
    assert config["quantization"] == config["quantization_config"] == {"bits": 4, "group_size": 64, "mode": "affine"}
    src_index = json.loads((a.source / "model.safetensors.index.json").read_text())
    dst_index = json.loads((a.model / "model.safetensors.index.json").read_text())
    manifest = {x["name"]: x for x in download["files"]}
    keys = {}
    source_keys = {}
    payload_bytes = 0
    sampled_matrices = 0
    errors = []
    unchanged = 0
    for record in conv["shards"]:
        src = a.source / record["source_file"]
        dst = a.model / record["file"]
        assert sha(src) == record["source_sha256"] == manifest[src.name]["sha256"]
        assert sha(dst) == record["sha256"]
        source, source_offset = header(src)
        target, target_offset = header(dst)
        source_keys.update({k: src.name for k in source})
        for key, spec in source.items():
            assert key not in keys, key
            assert spec["shape"] == expected_shapes[key], key
            if len(spec["shape"]) == 1:
                assert target[key] == spec or (target[key]["shape"] == spec["shape"] and target[key]["dtype"] == spec["dtype"])
                np.testing.assert_array_equal(rows(src, spec, source_offset, 1), rows(dst, target[key], target_offset, 1))
                keys[key] = dst.name
                unchanged += 1
                continue
            assert len(spec["shape"]) == 2 and spec["dtype"] == "BF16", key
            prefix = key.removesuffix(".weight")
            scale_key, bias_key = prefix + ".scales", prefix + ".biases"
            height, width = spec["shape"]
            assert target[key]["shape"] == [height, width // 8] and target[key]["dtype"] == "U32"
            for k in [scale_key, bias_key]:
                assert target[k]["shape"] == [height, width // 64] and target[k]["dtype"] == "BF16", k
            packed = rows(dst, target[key], target_offset)
            integers = ((packed[..., None] >> (np.arange(8, dtype=np.uint32) * 4)) & 15).reshape(-1, width)
            scales = rows(dst, target[scale_key], target_offset)
            biases = rows(dst, target[bias_key], target_offset)
            rebuilt = (integers.reshape(-1, width // 64, 64).astype(np.float32) * scales[..., None] + biases[..., None]).reshape(-1, width)
            reference = rows(src, spec, source_offset)
            assert np.isfinite(rebuilt).all() and np.isfinite(reference).all(), key
            expected_integers, expected_scales, expected_biases = independent_affine4(reference)
            np.testing.assert_array_equal(integers, expected_integers, err_msg=key)
            np.testing.assert_array_equal(scales.ravel(), expected_scales, err_msg=scale_key)
            np.testing.assert_array_equal(biases.ravel(), expected_biases, err_msg=bias_key)
            error = np.abs(rebuilt - reference)
            errors.append({"name": key, "sample_max_abs_error": float(error.max())})
            sampled_matrices += 1
            for k in [key, scale_key, bias_key]:
                keys[k] = dst.name
        assert set(target) == {k for k, v in keys.items() if v == dst.name}
        payload_bytes += dst.stat().st_size - target_offset
    assert source_keys == src_index["weight_map"]
    assert set(source_keys) == set(expected_shapes)
    assert keys == dst_index["weight_map"]
    assert payload_bytes == dst_index["metadata"]["total_size"]
    copied = []
    for name in ["tokenizer.json", "tokenizer.model", "tokenizer_config.json", "added_tokens.json", "special_tokens_map.json", "generation_config.json"]:
        assert sha(a.source / name) == sha(a.model / name) == manifest[name]["sha256"]
        copied.append(name)
    result = {"status": "pass", "source": conv["source"], "revision": conv["revision"],
              "full_model_inference_tested": False, "source_tensor_count": len(source_keys),
              "source_parameter_count": sum(math.prod(s) for s in expected_shapes.values()),
              "output_tensor_count": len(keys), "weight_payload_bytes": payload_bytes,
              "sampled_quantized_matrices": sampled_matrices, "unchanged_norms": unchanged,
              "copied_metadata_exact": copied, "quantization_samples": errors,
              "scope": "Complete header/shape/SHA audit; first two rows per matrix unpacked and quantization codes/scales/biases independently reproduced with NumPy",
              "quantizer_reference": "https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/cpu/quantized.cpp"}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "quantization_samples"}))


if __name__ == "__main__":
    main()
