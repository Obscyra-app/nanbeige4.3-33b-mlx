"""Synthetic executable equivalence check against mlx-lm's converter."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm.models.llama import Model, ModelArgs
from mlx_lm.utils import quantize_model

from convert import convert_shard
from prepare import REVISION, REPO, atomic_json, digest


def main():
    mx.set_default_device(mx.cpu)
    mx.random.seed(43)
    config = dict(model_type="llama", hidden_size=64, num_hidden_layers=2,
                  intermediate_size=128, num_attention_heads=4,
                  num_key_value_heads=2, head_dim=16, rms_norm_eps=1e-5,
                  vocab_size=128, tie_word_embeddings=False, rope_theta=120000000)
    config.update(architectures=["LlamaForCausalLM"], hidden_act="silu", attention_bias=False, mlp_bias=False)
    model = Model(ModelArgs.from_dict(config))
    model.set_dtype(mx.bfloat16)
    original = dict(tree_flatten(model.parameters()))
    mx.eval(original)
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        source, target = root / "source", root / "converted"
        source.mkdir(); target.mkdir()
        mx.save_safetensors(str(source / "model.safetensors"), original)
        result = convert_shard(source / "model.safetensors", target / "model.safetensors")
        converted = mx.load(str(target / "model.safetensors"))
        model, _ = quantize_model(model, config, group_size=64, bits=4)
        expected = dict(tree_flatten(model.parameters()))
        mx.eval(expected)
        assert converted.keys() == expected.keys()
        for name in expected:
            assert converted[name].dtype == expected[name].dtype, name
            assert converted[name].shape == expected[name].shape, name
            assert bool(mx.array_equal(converted[name], expected[name])), name
        model.load_weights(list(converted.items()), strict=True)
        atomic_json(source / "config.json", config)
        q = {"bits": 4, "group_size": 64, "mode": "affine"}
        atomic_json(target / "config.json", {**config, "quantization": q, "quantization_config": q})
        atomic_json(source / "model.safetensors.index.json", {"weight_map": {k: "model.safetensors" for k in original}})
        atomic_json(target / "model.safetensors.index.json", {"weight_map": {k: "model.safetensors" for k in expected}, "metadata": {"total_size": result["tensor_bytes"]}})
        for name in ["tokenizer.json", "tokenizer.model", "tokenizer_config.json", "added_tokens.json", "special_tokens_map.json", "generation_config.json"]:
            (source / name).write_text("{}\n")
            shutil.copyfile(source / name, target / name)
        atomic_json(root / "download.json", {"source": REPO, "revision": REVISION, "complete": True, "files": [{"name": f.name, "sha256": digest(f), "bytes": f.stat().st_size} for f in source.iterdir()]})
        atomic_json(root / "conversion.json", {"source": REPO, "revision": REVISION, "complete": True, "bits": 4, "group_size": 64, "shards": [result]})
        subprocess.check_call([sys.executable, str(Path(__file__).with_name("audit_package.py")),
                               "--source", str(source), "--model", str(target),
                               "--conversion-receipt", str(root / "conversion.json"),
                               "--download-receipt", str(root / "download.json"),
                               "--output", str(root / "audit.json")])
        print(json.dumps({"status": "pass", "fixture": "synthetic two-layer Llama, no pretrained inference",
                          "exact_tensor_matches": len(expected),
                          "quantized_matrices": len(result["quantized"]),
                          "unchanged_norms": len(result["unchanged"]),
                          "strict_load": True, "independent_numpy_audit": "pass"}))


if __name__ == "__main__":
    main()
