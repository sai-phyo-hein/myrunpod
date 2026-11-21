#!/usr/bin/env python3
"""
RunPod Serverless handler for VoxCPM2 text-to-speech.

  ENGINE=nano     nanovllm-voxcpm: ~3.5x faster, serves several requests at once per worker
  ENGINE=default  reference VoxCPM: slower, one request at a time

Modes (https://voxcpm.readthedocs.io/en/latest/usage_guide.html#three-generation-modes):
  design  voice from a text description, no reference audio
  clone   timbre from reference audio, style from an optional instruction
  hifi    reference audio + its exact transcript, highest similarity

Request / response format: see README.md. Local test: python handler.py (reads test_input.json).
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import inspect
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.parse
import urllib.request
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

PROCESS_START = time.time()  # worker (container) start, before heavy imports

import numpy as np
import runpod
import soundfile as sf

# --------------------------------------------------------------------------- config
ENGINE_NAME = os.getenv("ENGINE", "nano").strip().lower()
MODEL_ID = os.getenv("MODEL_ID", "openbmb/VoxCPM2")
MAX_CONCURRENCY = max(1, int(os.getenv("MAX_CONCURRENCY", "8")))   # jobs per worker (nano)
MAX_NUM_SEQS = int(os.getenv("MAX_NUM_SEQS", "16"))                # nano scheduler slots
GPU_MEM_UTIL = float(os.getenv("GPU_MEM_UTIL", "0.85"))
VOICES_DIR = Path(os.getenv("VOICES_DIR", "/app/voices"))
MAX_TEXT_CHARS = int(os.getenv("MAX_TEXT_CHARS", "3000"))
MAX_SEGMENT_CHARS = int(os.getenv("MAX_SEGMENT_CHARS", "250"))
SEGMENT_GAP_S = float(os.getenv("SEGMENT_GAP_S", "0.15"))
MAX_REF_BYTES = int(os.getenv("MAX_REF_MB", "20")) * 1024 * 1024
MAX_REF_SECONDS = float(os.getenv("MAX_REF_SECONDS", "60"))
ALLOW_REF_URL = os.getenv("ALLOW_REF_URL", "1") == "1"
LATENT_CACHE_SIZE = int(os.getenv("LATENT_CACHE_SIZE", "32"))
RETRY_BADCASE = os.getenv("RETRY_BADCASE", "1") == "1"            # default engine only
SMOKE_TEST = os.getenv("SMOKE_TEST", "1") == "1"
REFRESH_ON_ERROR = os.getenv("REFRESH_ON_ERROR", "1") == "1"
IMAGE_FA_ARCH = os.getenv("FLASH_ATTN_ARCH", "")                   # baked in by the Dockerfile
LOG_GEO = os.getenv("LOG_GEO", "1") == "1"                         # look up worker location once
RUNPOD_HF_CACHE = os.getenv("RUNPOD_HF_CACHE", "/runpod-volume/huggingface-cache/hub")

MODES = ("design", "clone", "hifi")
AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg")
OUTPUT_FORMATS = ("wav", "flac")


def log(*args):
    print("[handler]", *args, flush=True)


class RequestError(ValueError):
    """Bad input. Reported to the caller; the worker stays healthy."""


# --------------------------------------------------------------------------- worker info
# Identifying values only; never log the platform's secret variables (API keys, job URLs).
WORKER_ENV = {
    "worker_id": ("RUNPOD_POD_ID", "CONTAINER_ID"),            # RunPod, then Vast
    "datacenter": ("RUNPOD_DC_ID",),
    "endpoint_id": ("RUNPOD_ENDPOINT_ID",),
    "hostname": ("RUNPOD_POD_HOSTNAME", "HOSTNAME"),
    "public_ip": ("RUNPOD_PUBLIC_IP", "PUBLIC_IPADDR"),
    "gpu_count": ("RUNPOD_GPU_COUNT", "GPU_COUNT"),
}
WORKER: dict = {}


def collect_worker_info() -> dict:
    env = os.environ
    if any(k.startswith("RUNPOD_") for k in env):
        WORKER["platform"] = "runpod"
    elif "CONTAINER_API_KEY" in env or any(k.startswith("VAST_") for k in env):
        WORKER["platform"] = "vast"
    else:
        WORKER["platform"] = "local"
    for key, names in WORKER_ENV.items():
        val = next((env[n] for n in names if env.get(n)), None)
        if val:
            WORKER[key] = val
    try:  # nvidia-smi, not torch: a CUDA context here would take VRAM from the engine
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
        if out:
            name, mem, driver = (x.strip() for x in out[0].split(","))
            WORKER.update(gpu=name, gpu_memory=mem, driver=driver)
    except Exception as e:  # noqa: BLE001
        WORKER["gpu"] = f"unknown ({type(e).__name__})"
    return WORKER


def lookup_location():
    """Public IP geolocation, in a background thread so it never delays start-up."""
    try:
        with urllib.request.urlopen("https://ipinfo.io/json", timeout=3) as r:
            geo = json.loads(r.read())
        WORKER["location"] = ", ".join(x for x in (geo.get("city"), geo.get("region"), geo.get("country")) if x)
        WORKER.setdefault("public_ip", geo.get("ip"))
        log(f"worker location: {WORKER['location']} ({geo.get('org', '?')})")
    except Exception as e:  # noqa: BLE001
        log(f"worker location lookup failed: {type(e).__name__}: {e}")


def worker_summary() -> dict:
    """Short worker description added to every response."""
    keys = ("platform", "worker_id", "datacenter", "location", "gpu")
    return {k: WORKER[k] for k in keys if WORKER.get(k)}


# --------------------------------------------------------------------------- helpers
class Params:
    """Keyword arguments a callable accepts; **kwargs (or no signature) accepts all."""

    def __init__(self, fn):
        try:
            sig = inspect.signature(fn)
            self.names = set(sig.parameters)
            self.var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            self.text = str(sig)
        except (TypeError, ValueError):
            self.names, self.var_kw, self.text = set(), True, "<signature unavailable>"

    def __contains__(self, name):
        return self.var_kw or name in self.names


def resolve_model_dir() -> str:
    """Prefer RunPod's cached-model mount (no download, not billed); fall back to Hub."""
    if os.path.isdir(MODEL_ID):
        return MODEL_ID
    from huggingface_hub import snapshot_download

    if os.path.isdir(RUNPOD_HF_CACHE):
        try:
            path = snapshot_download(MODEL_ID, cache_dir=RUNPOD_HF_CACHE, local_files_only=True)
            log(f"Model from RunPod cache: {path}")
            return path
        except Exception as e:  # noqa: BLE001
            log(f"{MODEL_ID} not in RunPod cache ({type(e).__name__}); downloading instead")
    else:
        log("No RunPod model cache mounted; downloading. Set the endpoint's Model field to avoid this.")
    return snapshot_download(MODEL_ID)


def find_sample_rates(obj, path=""):
    found = []
    if isinstance(obj, dict):
        for key, val in obj.items():
            p = f"{path}.{key}" if path else key
            if isinstance(val, (int, float)) and ("sample_rate" in key.lower() or "sampling_rate" in key.lower()):
                found.append((p, int(val)))
            else:
                found.extend(find_sample_rates(val, p))
    elif isinstance(obj, list):
        for i, val in enumerate(obj):
            found.extend(find_sample_rates(val, f"{path}[{i}]"))
    return found


def detect_sample_rate(model_dir: str) -> int:
    if os.getenv("SAMPLE_RATE"):
        return int(os.environ["SAMPLE_RATE"])
    found = find_sample_rates(json.loads((Path(model_dir) / "config.json").read_text()))
    for hint in ("out", "decode"):
        for p, v in found:
            if hint in p.lower().split(".")[-1]:
                return v
    if not found:
        raise RuntimeError("No sample rate in config.json; set the SAMPLE_RATE env var.")
    return max(v for _, v in found)  # VoxCPM2 encodes at 16 kHz, decodes higher


def split_text(text: str, max_chars: int) -> list[str]:
    """Split long text at sentence ends (Burmese ။ included), packing up to max_chars."""
    if len(text) <= max_chars:
        return [text]
    parts = [p.strip() for p in re.split(r"(?<=[။!?！？。])\s*|(?<=\.)\s+", text) if p and p.strip()]
    segs, cur = [], ""
    for p in parts:
        while len(p) > max_chars:  # one very long sentence: break at a space if possible
            cut = p.rfind(" ", 0, max_chars)
            cut = cut if cut > max_chars // 2 else max_chars
            if cur:
                segs.append(cur)
                cur = ""
            segs.append(p[:cut].strip())
            p = p[cut:].strip()
        if not p:
            continue
        if cur and len(cur) + 1 + len(p) > max_chars:
            segs.append(cur)
            cur = p
        else:
            cur = f"{cur} {p}" if cur else p
    if cur:
        segs.append(cur)
    return segs


def join_segments(wavs: list[np.ndarray], sr: int) -> np.ndarray:
    wavs = [w for w in wavs if w.size]
    if len(wavs) <= 1:
        return wavs[0] if wavs else np.zeros(0, dtype=np.float32)
    gap = np.zeros(int(SEGMENT_GAP_S * sr), dtype=np.float32)
    out = []
    for i, w in enumerate(wavs):
        if i:
            out.append(gap)
        out.append(w)
    return np.concatenate(out)


def encode_audio(wav: np.ndarray, sr: int, fmt: str) -> str:
    buf = io.BytesIO()
    sf.write(buf, np.clip(wav, -1.0, 1.0), sr, format=fmt.upper(), subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# --------------------------------------------------------------------------- requests
@dataclass
class TTSRequest:
    text: str
    mode: str
    instruction: str
    ref_bytes: bytes | None
    ref_suffix: str
    prompt_text: str | None
    cfg_value: float
    inference_timesteps: int
    output_format: str
    split: bool
    warnings: list = field(default_factory=list)


def list_voices() -> list[str]:
    if not VOICES_DIR.is_dir():
        return []
    return sorted({p.stem for p in VOICES_DIR.iterdir() if p.suffix.lower() in AUDIO_EXTS})


def load_voice(name: str):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name):
        raise RequestError("voice names may only contain letters, digits, '_' and '-'")
    for ext in AUDIO_EXTS:
        p = VOICES_DIR / f"{name}{ext}"
        if p.exists():
            txt = VOICES_DIR / f"{name}.txt"
            transcript = txt.read_text(encoding="utf-8").strip() if txt.exists() else None
            return p.read_bytes(), ext, transcript
    raise RequestError(f"unknown voice '{name}'; available: {list_voices() or 'none'}")


def fetch_url(url: str):
    if not ALLOW_REF_URL:
        raise RequestError("reference_audio_url is disabled on this endpoint")
    if not url.startswith(("https://", "http://")):
        raise RequestError("reference_audio_url must be http(s)")
    req = urllib.request.Request(url, headers={"User-Agent": "voxcpm-runpod"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read(MAX_REF_BYTES + 1)
    except Exception as e:  # noqa: BLE001
        raise RequestError(f"could not download reference audio: {e}") from e
    if len(data) > MAX_REF_BYTES:
        raise RequestError(f"reference audio larger than {MAX_REF_BYTES // 2**20} MB")
    suffix = Path(urllib.parse.urlparse(url).path).suffix.lower()
    return data, suffix if suffix in AUDIO_EXTS else ".wav"


def check_reference(data: bytes, warnings: list):
    try:
        info = sf.info(io.BytesIO(data))
    except Exception as e:  # noqa: BLE001
        raise RequestError(f"reference audio is not a readable audio file: {e}") from e
    if info.duration > MAX_REF_SECONDS:
        raise RequestError(f"reference audio is {info.duration:.0f}s; max {MAX_REF_SECONDS:.0f}s")
    if not 5 <= info.duration <= 30:
        warnings.append(f"reference audio is {info.duration:.1f}s; 5-30 s gives the most stable cloning")


def parse_request(inp: dict) -> TTSRequest:
    warnings: list[str] = []
    text = str(inp.get("text", "")).strip()
    if not text:
        raise RequestError("'text' is required")
    if len(text) > MAX_TEXT_CHARS:
        raise RequestError(f"'text' is {len(text)} characters; max {MAX_TEXT_CHARS}")

    ref_bytes, ref_suffix = None, ".wav"
    prompt_text = inp.get("prompt_text")
    if inp.get("voice"):
        ref_bytes, ref_suffix, voice_txt = load_voice(str(inp["voice"]))
        prompt_text = prompt_text or voice_txt
    elif inp.get("reference_audio_base64"):
        try:
            ref_bytes = base64.b64decode(inp["reference_audio_base64"], validate=True)
        except Exception as e:  # noqa: BLE001
            raise RequestError("reference_audio_base64 is not valid base64") from e
        if len(ref_bytes) > MAX_REF_BYTES:
            raise RequestError(f"reference audio larger than {MAX_REF_BYTES // 2**20} MB")
        ref_suffix = "." + str(inp.get("reference_audio_format", "wav")).lower().lstrip(".")
    elif inp.get("reference_audio_url"):
        ref_bytes, ref_suffix = fetch_url(str(inp["reference_audio_url"]))
    if ref_bytes is not None:
        check_reference(ref_bytes, warnings)

    mode = str(inp.get("mode") or ("clone" if ref_bytes else "design")).lower()
    if mode not in MODES:
        raise RequestError(f"'mode' must be one of {MODES}")
    if mode in ("clone", "hifi") and ref_bytes is None:
        raise RequestError(f"mode '{mode}' needs 'voice', 'reference_audio_base64' or 'reference_audio_url'")
    if mode == "hifi" and not prompt_text:
        raise RequestError("mode 'hifi' needs 'prompt_text' (exact transcript of the reference audio)")
    if mode == "design" and ref_bytes is not None:
        warnings.append("reference audio ignored in design mode")
        ref_bytes = None

    instruction = str(inp.get("instruction") or "").strip()
    if instruction and not instruction.startswith("("):
        instruction = f"({instruction})"
    if mode == "hifi" and instruction:
        warnings.append("instruction ignored in hifi mode")
        instruction = ""

    try:
        cfg = float(inp.get("cfg_value", 2.0))
        steps = int(inp.get("inference_timesteps", 10))
    except (TypeError, ValueError) as e:
        raise RequestError("cfg_value must be a number and inference_timesteps an integer") from e
    if not 0.5 <= cfg <= 5.0:
        raise RequestError("cfg_value must be between 0.5 and 5.0 (typical 1.0-3.0)")
    if not 1 <= steps <= 50:
        raise RequestError("inference_timesteps must be between 1 and 50 (recommended 4-30)")

    fmt = str(inp.get("output_format", "wav")).lower()
    if fmt not in OUTPUT_FORMATS:
        raise RequestError(f"output_format must be one of {OUTPUT_FORMATS}")

    split = bool(inp.get("split_long_text", True))
    if split and mode == "design" and len(text) > MAX_SEGMENT_CHARS:
        # Each design call invents a new voice, so segments would not match.
        warnings.append("long text not split in design mode (voice would change per segment); "
                        "use clone/hifi for long text")
        split = False

    return TTSRequest(text=text, mode=mode, instruction=instruction, ref_bytes=ref_bytes,
                      ref_suffix=ref_suffix, prompt_text=prompt_text, cfg_value=cfg,
                      inference_timesteps=steps, output_format=fmt, split=split, warnings=warnings)


# --------------------------------------------------------------------------- smoke test
SMOKE_CODE = r"""
import json, os, torch
import torch.nn.functional as F
cap = torch.cuda.get_device_capability(0)
name = torch.cuda.get_device_name(0)
arch = os.environ.get("FLASH_ATTN_ARCH", "")
archs = [int(a) for a in arch.replace(";", " ").replace("-", " ").split() if a.isdigit()]
if archs and cap[0] not in {a // 10 for a in archs}:   # sm_80 kernels run on all 8.x GPUs, etc.
    print(json.dumps({"ok": False, "error": f"image built for sm_{arch}, but {name} is sm_{cap[0]}{cap[1]}; "
                      f"use an image tag that includes sm_{cap[0]}0 or restrict the endpoint's GPU types"}))
    raise SystemExit(0)
from flash_attn import flash_attn_varlen_func
torch.manual_seed(0)
n, h, d = 256, 8, 64
q, k, v = (torch.randn(n, h, d, device="cuda", dtype=torch.bfloat16) for _ in range(3))
cu = torch.tensor([0, n], dtype=torch.int32, device="cuda")
out = flash_attn_varlen_func(q, k, v, cu, cu, n, n, causal=True)
ref = F.scaled_dot_product_attention(q.transpose(0, 1)[None], k.transpose(0, 1)[None],
                                     v.transpose(0, 1)[None], is_causal=True)[0].transpose(0, 1)
err = (out - ref).abs().max().item()
print(json.dumps({"ok": err < 0.05, "gpu": name, "capability": list(cap), "max_err": err}))
"""


def run_smoke_test():
    """Run in a subprocess so its CUDA context is released before the engine starts."""
    t = time.perf_counter()
    r = subprocess.run([sys.executable, "-c", SMOKE_CODE], capture_output=True, text=True, timeout=300)
    lines = [ln for ln in r.stdout.strip().splitlines() if ln.startswith("{")]
    if r.returncode != 0 or not lines:
        raise RuntimeError(f"flash-attn smoke test crashed:\n{r.stderr[-2000:]}")
    res = json.loads(lines[-1])
    if not res.get("ok"):
        raise RuntimeError(f"flash-attn smoke test failed: {res}")
    log(f"flash-attn smoke test OK in {time.perf_counter() - t:.1f}s: {res}")


# --------------------------------------------------------------------------- engines
class NanoEngine:
    name = "nano"

    def __init__(self):
        self.server = None
        self.sample_rate = 0
        self.latents: OrderedDict[str, object] = OrderedDict()

    async def start(self):
        if SMOKE_TEST:
            await asyncio.to_thread(run_smoke_test)
        model_dir = await asyncio.to_thread(resolve_model_dir)
        self.sample_rate = detect_sample_rate(model_dir)
        from nanovllm_voxcpm import VoxCPM

        # Called inside the running event loop -> returns the async server pool.
        self.server = VoxCPM.from_pretrained(
            model=model_dir, devices=[0], max_num_seqs=MAX_NUM_SEQS, gpu_memory_utilization=GPU_MEM_UTIL
        )
        ready = getattr(self.server, "wait_for_ready", None)
        if ready:
            await ready()
        self.params = Params(self.server.generate)
        log(f"nano generate{self.params.text}")
        enc = getattr(self.server, "encode_latents", None)
        log(f"nano encode_latents{Params(enc).text if enc else ': MISSING (clone/hifi unavailable)'}")
        log(f"sample rate {self.sample_rate} Hz")

    async def _encode(self, data: bytes, suffix: str):
        fn = getattr(self.server, "encode_latents", None)
        if fn is None:
            raise RequestError("this nanovllm-voxcpm version cannot encode reference audio")
        fmt = suffix.lstrip(".") or "wav"
        errors = []
        for desc, args in (("bytes, format", (data, fmt)), ("bytes", (data,))):
            try:
                res = fn(*args)
                return await res if inspect.isawaitable(res) else res
            except Exception as e:  # noqa: BLE001 - try the next call shape
                errors.append(f"{desc}: {type(e).__name__}: {e}")
        fd, path = tempfile.mkstemp(suffix=suffix)
        try:
            os.write(fd, data)
            os.close(fd)
            res = fn(path)
            return await res if inspect.isawaitable(res) else res
        except Exception as e:  # noqa: BLE001
            errors.append(f"path: {type(e).__name__}: {e}")
        finally:
            os.unlink(path)
        raise RuntimeError("encode_latents failed:\n  " + "\n  ".join(errors))

    async def latents_for(self, data: bytes, suffix: str):
        key = hashlib.sha256(data).hexdigest()
        if key in self.latents:
            self.latents.move_to_end(key)
            return self.latents[key], True
        lat = await self._encode(data, suffix)
        self.latents[key] = lat
        while len(self.latents) > LATENT_CACHE_SIZE:
            self.latents.popitem(last=False)
        return lat, False

    async def _generate(self, text: str, kw: dict) -> np.ndarray:
        chunks = []
        async for chunk in self.server.generate(target_text=text, **kw):
            chunks.append(np.asarray(chunk, dtype=np.float32).reshape(-1))
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)

    async def synthesize(self, req: TTSRequest):
        kw = {k: v for k, v in {"cfg_value": req.cfg_value,
                                "inference_timesteps": req.inference_timesteps}.items() if k in self.params}
        info = {}
        if req.mode != "design":
            need = ["ref_audio_latents"] + (["prompt_latents", "prompt_text"] if req.mode == "hifi" else [])
            missing = [n for n in need if n not in self.params]
            if missing:
                raise RequestError(f"engine 'nano' does not support mode '{req.mode}' (missing {missing})")
            lat, hit = await self.latents_for(req.ref_bytes, req.ref_suffix)
            info["reference_cache_hit"] = hit
            kw["ref_audio_latents"] = lat
            if req.mode == "hifi":
                kw.update(prompt_latents=lat, prompt_text=req.prompt_text)
        segments = split_text(req.text, MAX_SEGMENT_CHARS) if req.split else [req.text]
        # Segments run concurrently; the scheduler batches them on the GPU.
        wavs = await asyncio.gather(*(self._generate(req.instruction + s, kw) for s in segments))
        return join_segments(list(wavs), self.sample_rate), len(segments), info


class VoxCPMEngine:
    name = "default"

    def __init__(self):
        self.model = None
        self.sample_rate = 0
        self.lock = asyncio.Lock()

    async def start(self):
        await asyncio.to_thread(self._load)

    def _load(self):
        model_dir = resolve_model_dir()
        from voxcpm import VoxCPM

        self.model = VoxCPM.from_pretrained(model_dir, load_denoiser=False)
        self.sample_rate = int(self.model.tts_model.sample_rate)
        log(f"VoxCPM loaded; sample rate {self.sample_rate} Hz")

    async def synthesize(self, req: TTSRequest):
        async with self.lock:
            return await asyncio.to_thread(self._synthesize, req)

    def _synthesize(self, req: TTSRequest):
        path = None
        try:
            kw = dict(cfg_value=req.cfg_value, inference_timesteps=req.inference_timesteps,
                      retry_badcase=RETRY_BADCASE, normalize=False, denoise=False)
            if req.ref_bytes is not None:
                fd, path = tempfile.mkstemp(suffix=req.ref_suffix)
                os.write(fd, req.ref_bytes)
                os.close(fd)
                kw["reference_wav_path"] = path
                if req.mode == "hifi":
                    kw.update(prompt_wav_path=path, prompt_text=req.prompt_text)
            segments = split_text(req.text, MAX_SEGMENT_CHARS) if req.split else [req.text]
            wavs = [np.asarray(self.model.generate(text=req.instruction + s, **kw), dtype=np.float32).reshape(-1)
                    for s in segments]
            return join_segments(wavs, self.sample_rate), len(segments), {}
        finally:
            if path:
                os.unlink(path)


# --------------------------------------------------------------------------- lifecycle
_engine = None
_engine_lock: asyncio.Lock | None = None


async def get_engine():
    """Load the engine once per worker, on the first job (the async nano server must be
    created inside RunPod's event loop). Send {"warmup": true} to pre-load."""
    global _engine, _engine_lock
    if _engine is not None:
        return _engine
    if _engine_lock is None:
        _engine_lock = asyncio.Lock()
    async with _engine_lock:
        if _engine is None:
            t = time.perf_counter()
            eng = NanoEngine() if ENGINE_NAME == "nano" else VoxCPMEngine()
            await eng.start()
            _engine = eng
            log(f"Engine '{eng.name}' ready in {time.perf_counter() - t:.1f}s "
                f"({time.time() - PROCESS_START:.1f}s after worker start)")
    return _engine


async def handler(job):
    inp = job.get("input") or {}
    try:
        if inp.get("warmup"):
            t = time.perf_counter()
            eng = await get_engine()
            return {"status": "warm", "engine": eng.name, "load_s": round(time.perf_counter() - t, 2),
                    "voices": list_voices(), "worker": worker_summary()}
        req = await asyncio.to_thread(parse_request, inp)
        t_load = time.perf_counter()
        engine = await get_engine()
        load_s = time.perf_counter() - t_load
        t0 = time.perf_counter()
        wav, n_segments, info = await engine.synthesize(req)
        gen_s = time.perf_counter() - t0
    except RequestError as e:
        return {"error": str(e)}
    except Exception as e:  # noqa: BLE001 - engine failure: report, then recycle the worker
        log(traceback.format_exc())
        out = {"error": f"{type(e).__name__}: {e}", "worker": worker_summary()}
        if REFRESH_ON_ERROR:
            out["refresh_worker"] = True
        return out

    if wav.size == 0:
        return {"error": "no audio was generated"}
    duration = wav.shape[0] / engine.sample_rate
    return {
        "audio_base64": encode_audio(wav, engine.sample_rate, req.output_format),
        "format": req.output_format,
        "sample_rate": engine.sample_rate,
        "duration_s": round(duration, 3),
        "generation_s": round(gen_s, 3),
        "rtf": round(gen_s / duration, 4),
        "engine": engine.name,
        "mode": req.mode,
        "segments": n_segments,
        "cold_start_load_s": round(load_s, 2) if load_s > 0.5 else 0.0,
        **info,
        "warnings": req.warnings,
        "worker": worker_summary(),
    }


def concurrency_modifier(current_concurrency: int) -> int:
    return MAX_CONCURRENCY if ENGINE_NAME == "nano" else 1


if __name__ == "__main__":
    if ENGINE_NAME not in ("nano", "default"):
        sys.exit(f"ENGINE must be 'nano' or 'default', got '{ENGINE_NAME}'")
    log(f"worker start: {json.dumps(collect_worker_info())}")
    platform_vars = sorted(k for k in os.environ if k.startswith(("RUNPOD_", "VAST_")))
    log(f"platform variables present (names only): {platform_vars}")
    if LOG_GEO:
        threading.Thread(target=lookup_location, daemon=True).start()
    log(f"engine={ENGINE_NAME} model={MODEL_ID} concurrency={concurrency_modifier(0)} "
        f"image_arch=sm_{IMAGE_FA_ARCH or '?'} voices={list_voices()}")
    log(f"job {job.get('id', '?')}: mode={req.mode} chars={len(req.text)} "
        f"segments={n_segments} audio={duration:.2f}s gen={gen_s:.2f}s")
    runpod.serverless.start({"handler": handler, "concurrency_modifier": concurrency_modifier})