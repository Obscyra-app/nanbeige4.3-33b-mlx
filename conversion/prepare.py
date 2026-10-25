"""Pinned, resumable public-source acquisition. Never logs redirected URLs."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.request

REPO = "Nanbeige/Nanbeige4.3-33B"
REVISION = "c1093c501cc6460427c0ac634db7223d0450c0ee"


def uncached_reader(file):
    """Avoid evicting application RAM while scanning many GB on macOS."""
    if sys.platform == "darwin":
        import fcntl
        fcntl.fcntl(file.fileno(), fcntl.F_NOCACHE, 1)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        uncached_reader(f)
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def download(root, entry, xet=False, authenticated=False):
    name = entry["rfilename"]
    path = root / name
    expected = entry.get("lfs", {}).get("sha256")
    size = entry["size"]
    if path.exists():
        actual = digest(path)
        if path.stat().st_size != size or (expected and actual != expected):
            raise ValueError(f"Existing source failed verification: {name}")
        return {"name": name, "bytes": size, "sha256": actual}
    if xet:
        from huggingface_hub import hf_hub_download
        started = time.monotonic()
        hf_hub_download(REPO, name, revision=REVISION, local_dir=root, token=True if authenticated else False)
        actual = digest(path)
        if path.stat().st_size != size or (expected and actual != expected):
            raise ValueError(f"Downloaded source failed verification: {name}")
        result = {"name": name, "bytes": size, "sha256": actual, "elapsed_seconds": time.monotonic() - started}
        print(json.dumps(result), flush=True)
        return result
    part = path.with_suffix(path.suffix + ".partial")
    url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}"
    started = time.monotonic()
    for attempt in range(6):
        offset = part.stat().st_size if part.exists() else 0
        if offset == size:
            break
        req = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                if offset and response.status != 206:
                    raise ValueError("Server did not honor resume range")
                if offset and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise ValueError("Incorrect resume range")
                with part.open("ab" if offset else "wb") as f:
                    for chunk in iter(lambda: response.read(4 * 1024 * 1024), b""):
                        f.write(chunk)
            if part.stat().st_size == size:
                break
        except Exception as exc:
            print(json.dumps({"retry": name, "attempt": attempt + 1, "error_type": type(exc).__name__}), flush=True)
            if attempt == 5:
                raise RuntimeError(f"Download failed: {name}") from None
            time.sleep(2)
    actual = digest(part)
    if part.stat().st_size != size or (expected and actual != expected):
        raise ValueError(f"Downloaded source failed verification: {name}")
    part.replace(path)
    result = {"name": name, "bytes": size, "sha256": actual, "elapsed_seconds": time.monotonic() - started}
    print(json.dumps(result), flush=True)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--receipt", type=Path, required=True)
    p.add_argument("--metadata-only", action="store_true")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--xet", action="store_true")
    p.add_argument("--authenticated", action="store_true", help="Use existing HF login; never copies or prints its token")
    p.add_argument("--start-shard", type=int, default=1)
    a = p.parse_args()
    a.source.mkdir(parents=True, exist_ok=True)
    a.receipt.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true", timeout=60) as f:
        model = json.load(f)
    if model["sha"] != REVISION:
        raise ValueError("Revision mismatch")
    files = [x for x in model["siblings"] if "/" not in x["rfilename"] and x["rfilename"] != ".gitattributes"]
    if a.metadata_only:
        files = [x for x in files if not x["rfilename"].endswith(".safetensors")]
    elif a.start_shard > 1:
        files = [x for x in files if x["rfilename"].endswith(".safetensors") and int(x["rfilename"].split("-")[1]) >= a.start_shard]
    receipt = {"source": REPO, "revision": REVISION, "files": [], "complete": False}
    atomic_json(a.receipt, receipt)
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures = [pool.submit(download, a.source, x, a.xet, a.authenticated) for x in files]
        for f in concurrent.futures.as_completed(futures):
            receipt["files"].append(f.result())
            atomic_json(a.receipt, receipt)
    receipt["complete"] = True
    receipt["metadata_only"] = a.metadata_only
    receipt["total_bytes"] = sum(x["bytes"] for x in receipt["files"])
    atomic_json(a.receipt, receipt)
    print(json.dumps({"complete": True, "files": len(files), "bytes": receipt["total_bytes"]}), flush=True)


if __name__ == "__main__":
    main()
