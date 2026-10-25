"""Exercise original template options without loading any model weights."""
import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--converted", type=Path)
    a = p.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(a.source, local_files_only=True, trust_remote_code=False)
    config = json.loads((a.source / "config.json").read_text())
    assert tokenizer.bos_token_id == config["bos_token_id"]
    assert tokenizer.eos_token_id == config["eos_token_id"]
    message = [{"role": "user", "content": "What is 2 + 2?"}]
    history = message + [{"role": "assistant", "content": "4", "reasoning_content": "Adding two and two."}, {"role": "user", "content": "And 3 + 3?"}]
    tools = [{"type": "function", "function": {"name": "add", "description": "Add two integers.", "parameters": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"]}}}]
    calls = message + [{"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "add", "arguments": {"a": 2, "b": 2}}}]}, {"role": "tool", "content": "4"}]
    fixtures = [("thinking", message, {"enable_thinking": True}),
                ("no_thinking", message, {"enable_thinking": False}),
                ("preserve_thinking", history, {"preserve_thinking": True}),
                ("remove_thinking", history, {"preserve_thinking": False}),
                ("tool_xml", calls, {"tools": tools, "tool_call_format": "xml"}),
                ("tool_json", calls, {"tools": tools, "tool_call_format": "json"})]
    converted = AutoTokenizer.from_pretrained(a.converted, local_files_only=True, trust_remote_code=False) if a.converted else None
    rendered = {}
    receipts = []
    for name, messages, kwargs in fixtures:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)
        ids = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False, add_generation_prompt=True, **kwargs)
        if isinstance(ids, dict):
            ids = ids["input_ids"]
        assert len(ids) > 0 and max(ids) < config["vocab_size"]
        if converted:
            assert text == converted.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)
            other = converted.apply_chat_template(messages, tokenize=True, return_dict=False, add_generation_prompt=True, **kwargs)
            if isinstance(other, dict):
                other = other["input_ids"]
            assert ids == other
        rendered[name] = text
        receipts.append({"fixture": name, "prompt_sha256": hashlib.sha256(text.encode()).hexdigest(), "token_count": len(ids)})
    assert rendered["thinking"].endswith("<think>\n")
    assert rendered["no_thinking"].endswith("<think>\n\n</think>\n\n")
    assert "Adding two and two." in rendered["preserve_thinking"] and "Adding two and two." not in rendered["remove_thinking"]
    assert "<function=add>" in rendered["tool_xml"]
    assert '"name": "add"' in rendered["tool_json"]
    result = {"status": "pass", "full_model_inference_tested": False, "converted_tokenizer_compared": bool(converted), "fixtures": receipts}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
