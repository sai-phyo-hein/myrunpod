#!/usr/bin/env python3
"""
Call the VoxCPM2 RunPod endpoint and save the audio. Standard library only.

  export RUNPOD_API_KEY=...            # RunPod console > Settings > API Keys
  export RUNPOD_ENDPOINT_ID=...        # shown on the endpoint page

  python client.py --text "မင်္ဂလာပါ။" --instruction "A young woman, warm voice" -o design.wav
  python client.py --text-file story.txt --voice Grey -o clone.wav
  python client.py --text-file story.txt --voice Grey --mode hifi -o hifi.wav
  python client.py --text "..." --ref my_voice.wav -o mine.wav       # upload your own reference
  python client.py --warmup                                           # pre-load a worker
"""
import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request

API = "https://api.runpod.ai/v2"


def call(url, key, payload=None, timeout=330):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, method="POST" if data else "GET",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                 "User-Agent": "voxcpm-client/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        sys.exit(f"HTTP {e.code}: {e.read().decode(errors='replace')[:500]}")


def main():
    ap = argparse.ArgumentParser(description="VoxCPM2 RunPod client")
    ap.add_argument("--endpoint", default=os.getenv("RUNPOD_ENDPOINT_ID"))
    ap.add_argument("--api-key", default=os.getenv("RUNPOD_API_KEY"))
    ap.add_argument("--text")
    ap.add_argument("--text-file")
    ap.add_argument("--mode", choices=["design", "clone", "hifi"])
    ap.add_argument("--instruction", help='e.g. "slightly faster, cheerful tone"')
    ap.add_argument("--voice", help="preset voice baked into the image, e.g. Grey")
    ap.add_argument("--ref", help="local reference audio file to upload")
    ap.add_argument("--ref-url", help="reference audio URL the worker downloads")
    ap.add_argument("--prompt-text", help="transcript of the reference (hifi)")
    ap.add_argument("--cfg", type=float)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--format", choices=["wav", "flac"], default="wav")
    ap.add_argument("--warmup", action="store_true")
    ap.add_argument("-o", "--output", default="out.wav")
    args = ap.parse_args()

    if not args.endpoint or not args.api_key:
        sys.exit("Set RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY (or pass --endpoint / --api-key).")

    if args.warmup:
        inp = {"warmup": True}
    else:
        text = args.text or (open(args.text_file, encoding="utf-8").read().strip() if args.text_file else None)
        if not text:
            sys.exit("Pass --text or --text-file.")
        inp = {"text": text, "output_format": args.format}
        for key, val in [("mode", args.mode), ("instruction", args.instruction), ("voice", args.voice),
                         ("reference_audio_url", args.ref_url), ("prompt_text", args.prompt_text),
                         ("cfg_value", args.cfg), ("inference_timesteps", args.steps)]:
            if val is not None:
                inp[key] = val
        if args.ref:
            with open(args.ref, "rb") as f:
                inp["reference_audio_base64"] = base64.b64encode(f.read()).decode()
            inp["reference_audio_format"] = os.path.splitext(args.ref)[1].lstrip(".") or "wav"

    t0 = time.perf_counter()
    res = call(f"{API}/{args.endpoint}/runsync", args.api_key, {"input": inp})
    while res.get("status") in ("IN_QUEUE", "IN_PROGRESS"):   # runsync returned before finishing
        time.sleep(1)
        res = call(f"{API}/{args.endpoint}/status/{res['id']}", args.api_key)
    wall = time.perf_counter() - t0

    out = res.get("output") or {}
    if res.get("status") != "COMPLETED" or "error" in out:
        sys.exit(f"Job {res.get('status')}: {out.get('error') or res.get('error') or res}")

    if args.warmup:
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return

    with open(args.output, "wb") as f:
        f.write(base64.b64decode(out["audio_base64"]))
    queue_s = res.get("delayTime", 0) / 1000
    exec_s = res.get("executionTime", 0) / 1000
    print(f"Saved {args.output}  ({out['duration_s']:.2f}s audio @ {out['sample_rate']} Hz, "
          f"{out['engine']} / {out['mode']}, {out['segments']} segment(s))")
    print(f"  generation {out['generation_s']:.2f}s | RTF {out['rtf']:.3f}"
          + (f" | cold-start load {out['cold_start_load_s']:.1f}s" if out.get("cold_start_load_s") else ""))
    print(f"  RunPod queue {queue_s:.2f}s | execution {exec_s:.2f}s | client round trip {wall:.2f}s")
    if "reference_cache_hit" in out:
        print(f"  reference latents cached: {out['reference_cache_hit']}")
    for w in out.get("warnings", []):
        print(f"  warning: {w}")


if __name__ == "__main__":
    main()
