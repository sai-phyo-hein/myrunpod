"""VoxCPM2 text-to-speech worker for RunPod Serverless (queue-based endpoint).

Input  (POST /runsync or /run, body {"input": {...}}):
  action            "ping" returns worker info without generating (use it to measure cold start)
  text              one sentence to synthesize            -> single-clip mode
  script            a paragraph, auto-split into sentences -> lesson mode
  sentences         explicit list of sentences             -> lesson mode
  mode              "design" | "clone" | "hifi" (default: clone if a reference is sent, else design)
                      design  voice from the style description alone; any reference is ignored
                      clone   timbre from the reference clip, style steers emotion/pace
                      hifi    reference clip + its exact transcript (prompt_text): closest match; style is ignored
  prompt_text       exact transcript of the reference clip (required for hifi; alias: reference_text)
  style             voice description (design) or style (clone), e.g. "A gentle, warm woman reading a bedtime story ..."
  steps             diffusion steps, 2-30 (default 10; fewer = faster, rougher)
  cfg               guidance scale, 0.5-5 (default 2.0)
  seed              integer, optional
  retry_badcase     bool (default false)
  reference_b64     base64 of a reference voice clip (wav/mp3/flac/ogg), optional
  reference_url     alternative to reference_b64: a (presigned) https URL of the clip, e.g. in R2. The worker downloads it
                    once and keeps it on local disk. Send reference_id (the voice name) and reference_version (R2 ETag)
                    too so a re-uploaded voice is fetched again while an unchanged one is reused.
  gap_s             silence between sentences in lesson mode (default 0.7)
  out_sample_rate   optional resample, e.g. 24000 (default: model rate, 48000)
  return_clips      lesson mode: also return each sentence separately (default true)

Sieng web app mode (the app's Vercel API sends this shape, no change needed on its side):
  {"input": {"method": "POST", "path": "/api/generate", "body": {text, speed, reference_id, reference_url,
   reference_version, group_id, project_id, process_id, ...}}}
  Optional body.mode and body.prompt_text select the generation mode as above; without them, nothing changes.
  The clip is generated with the reference voice, uploaded to R2 as body.output_key when given (a plain relative .wav path),
  otherwise outputs/<group_id>/<project_id>/<process_id>.wav
  and the handler returns {"audio_url": <presigned https URL>, "process_id": ..., "duration_s": ...}, which RunPod delivers as output.audio_url.
  Needs R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME (optional R2_ENDPOINT, VOX_OUT_URL_TTL_S).
  Long text is split into sentences (max VOX_MAX_CHARS each, up to VOX_MAX_SENTENCES pieces) and joined with short pauses.
  "speed" below 0.85 / above 1.1 adds "speaking slowly" / "a little faster" to the style; "style" in the body is used as given.

Environment variables:
  VOX_MODEL_DIR     baked-in weights, default /models/VoxCPM2
  CFG (VOX_CFG)    default cfg_value, 2.0 if unset; 3.0 makes the style clearer. A request's own "cfg" wins.
  VOX_MODEL_SOURCE  auto (default: RunPod cached model if present, else baked) | cache | baked
  VOX_HF_ID         Hugging Face id used to find the RunPod cached copy, default openbmb/VoxCPM2
  VOX_DTYPE         float16 (default) or bfloat16
  VOX_OPTIMIZE      0 (default) or 1 (torch.compile; slower start, unclear benefit)
  VOX_MAX_CHARS     max characters per sentence (default 400)
  VOX_REF_CACHE     how many encoded reference voices to keep in memory (default 8)
  VOX_MAX_SENTENCES max sentences per request (default 12)
  VOX_FAKE          1 = stand-in engine that makes tones (for testing the plumbing without a GPU)
"""
import time as _time
_T0 = _time.perf_counter()  # roughly "Python started"; used for boot_to_ready_s

import base64
import collections
import hashlib
import io
import json
import os
import re
import tempfile
import time
import traceback
import zlib

import numpy as np
import soundfile as sf
import runpod

FAKE = os.environ.get("VOX_FAKE", "0") == "1"
MODEL_DIR = os.environ.get("VOX_MODEL_DIR", "/models/VoxCPM2")
MODEL_SOURCE = os.environ.get("VOX_MODEL_SOURCE", "auto").lower()
HF_ID = os.environ.get("VOX_HF_ID", "openbmb/VoxCPM2")
CACHE_ROOT = os.environ.get("VOX_CACHE_ROOT", "/runpod-volume/huggingface-cache/hub")
DTYPE = os.environ.get("VOX_DTYPE", "float16")
OPTIMIZE = os.environ.get("VOX_OPTIMIZE", "0") == "1"
MAX_CHARS = int(os.environ.get("VOX_MAX_CHARS", "400"))
MAX_SENTENCES = int(os.environ.get("VOX_MAX_SENTENCES", "12"))
REF_CACHE_SIZE = int(os.environ.get("VOX_REF_CACHE", "8"))
MAX_REF_BYTES = int(os.environ.get("VOX_MAX_REF_BYTES", str(15 * 1024 * 1024)))
REF_DIR = os.environ.get("VOX_REF_DIR", "/tmp/vox_refs")
# Default seed for every request that does not send its own (VOX_SEED, or just SEED). Empty = random each time.
DEFAULT_SEED = (os.environ.get("VOX_SEED") or os.environ.get("SEED") or "").strip()
# cfg_value = how strongly the model follows the text and the style. 2.0 is VoxCPM's default; 3.0 gives clearer styles
# (the test notebook uses 3.0) but can buzz on long text. Per-request "cfg" in the body still wins.
try:
    DEFAULT_CFG = float((os.environ.get("VOX_CFG") or os.environ.get("CFG") or "2.0").strip())
except ValueError:
    raise RuntimeError("CFG / VOX_CFG must be a number, e.g. 3.0")
REF_DOWNLOAD_TIMEOUT_S = float(os.environ.get("VOX_REF_TIMEOUT_S", "20"))
REF_ALLOW_HTTP = os.environ.get("VOX_ALLOW_HTTP", "0") == "1"           # tests only
REF_HOSTS = [h.strip().lower() for h in os.environ.get("VOX_REF_HOSTS", "").split(",") if h.strip()]  # e.g. ".r2.cloudflarestorage.com"
WARM_TEXT = "မင်္ဂလာပါ။"
MODES = ("design", "clone", "hifi")
MAX_PROMPT_TEXT_CHARS = int(os.environ.get("VOX_MAX_PROMPT_TEXT", "1000"))

STATE = {"jobs": 0, "load_s": None, "warmup_s": None, "gpu": None}
ENGINE = None


class UserError(Exception):
    """Bad request: reported back to the caller as {"error": ...}."""


# --------------------------------------------------------------------------- engines
def resolve_cached_snapshot(model_id):
    """Find a model in RunPod's cached-models location (Hugging Face cache layout), or None."""
    org, name = model_id.split("/", 1)
    root = os.path.join(CACHE_ROOT, f"models--{org}--{name}")
    refs = os.path.join(root, "refs", "main")
    snaps = os.path.join(root, "snapshots")
    if os.path.isfile(refs):
        p = os.path.join(snaps, open(refs).read().strip())
        if os.path.isdir(p):
            return p
    if os.path.isdir(snaps):
        for d in sorted(os.listdir(snaps)):
            p = os.path.join(snaps, d)
            if os.path.isdir(p):
                return p
    return None


def pick_model_dir():
    """Return (directory, source) where source is 'runpod-cache' or 'baked'."""
    if MODEL_SOURCE not in ("auto", "cache", "baked"):
        raise RuntimeError(f"VOX_MODEL_SOURCE must be auto, cache or baked (got {MODEL_SOURCE!r})")
    if MODEL_SOURCE in ("auto", "cache"):
        p = resolve_cached_snapshot(HF_ID)
        if p and os.path.exists(os.path.join(p, "config.json")):
            return p, "runpod-cache"
        if MODEL_SOURCE == "cache":
            raise RuntimeError(f"VOX_MODEL_SOURCE=cache but no cached copy of {HF_ID} under {CACHE_ROOT}. "
                               f"Set the endpoint's 'Model' to {HF_ID} and save.")
    if os.path.exists(os.path.join(MODEL_DIR, "config.json")):
        return MODEL_DIR, "baked"
    raise RuntimeError(f"No model found: nothing baked at {MODEL_DIR} and no RunPod cached copy of {HF_ID}. "
                       f"Either set the endpoint's 'Model' to {HF_ID}, or build the image with BAKE_MODEL=1.")


def prepare_model_dir(src_dir, source):
    """Return a model directory whose config.json has the requested dtype.
    Large files are symlinked, so this is instant (no multi-GB copy)."""
    cfg = json.load(open(os.path.join(src_dir, "config.json")))
    if cfg.get("dtype") == DTYPE:
        return src_dir
    d = f"/tmp/model_{DTYPE}_{source}"
    os.makedirs(d, exist_ok=True)
    for name in os.listdir(src_dir):
        dst = os.path.join(d, name)
        if name != "config.json" and not os.path.lexists(dst):
            os.symlink(os.path.join(src_dir, name), dst)
    cfg["dtype"] = DTYPE
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    return d


class RealEngine:
    def __init__(self):
        t = time.perf_counter()
        import torch
        import check_env
        check_env.check_package_versions()
        if torch.cuda.is_available():
            check_env.check_gpu_kernels()
        if OPTIMIZE:
            try:
                import triton  # noqa: F401
            except ImportError:
                raise RuntimeError("VOX_OPTIMIZE=1 needs triton, which this image removes to save about 565 MB. "
                                   "Rebuild with --build-arg KEEP_TRITON=1, or set VOX_OPTIMIZE=0.")
        from voxcpm import VoxCPM
        self.import_s = round(time.perf_counter() - t, 1)
        self._ref_cache = collections.OrderedDict()
        self.ref_encodes = 0
        self.ref_hits = 0
        self.fast_path = True       # clone mode: reuse an encoded reference
        self.fast_path_hifi = True  # hifi mode: reuse an encoded reference + transcript

        src_dir, self.source = pick_model_dir()
        t = time.perf_counter()
        self.model = VoxCPM.from_pretrained(prepare_model_dir(src_dir, self.source), load_denoiser=False, optimize=OPTIMIZE)
        self.weights_s = round(time.perf_counter() - t, 1)
        cls = type(self.model.tts_model).__name__
        if cls != "VoxCPM2Model":
            raise RuntimeError(f"Loaded {cls}, expected VoxCPM2Model. Wrong voxcpm version or wrong checkpoint.")
        self.sr = self.model.tts_model.sample_rate
        self.gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        self.versions = check_env.versions()
        self.arch = (lambda c: f"sm_{c[0]}{c[1]}")(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None

    # voxcpm's generate() has no `seed` argument (an earlier version of this handler passed one and would have
    # failed on any request that set it), so seed the random generators directly.
    def _seed(self, seed):
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _get_prompt_cache(self, ref_path, ref_key, prompt_text=None):
        """Encode a reference clip once and reuse it. model.generate() re-reads and re-encodes the clip on
        every call; the prompt cache it builds internally is read-only during generation, so it can be shared.
        Hi-Fi caches are keyed by the transcript too, since the same clip with another transcript differs."""
        key = (ref_key, hashlib.sha1(prompt_text.encode("utf-8")).hexdigest() if prompt_text else None)
        if key in self._ref_cache:
            self._ref_cache.move_to_end(key)
            self.ref_hits += 1
            return self._ref_cache[key]
        kw = {"reference_wav_path": ref_path}
        if prompt_text:  # Hi-Fi: the clip is also the prompt the model continues from
            kw.update(prompt_wav_path=ref_path, prompt_text=prompt_text)
        cache = self.model.tts_model.build_prompt_cache(**kw)
        self.ref_encodes += 1
        self._ref_cache[key] = cache
        while len(self._ref_cache) > REF_CACHE_SIZE:
            self._ref_cache.popitem(last=False)
        return cache

    def generate(self, text, steps, cfg, seed, retry, ref_path, ref_key=None, prompt_text=None):
        """design: no ref_path. clone: ref_path. hifi: ref_path + prompt_text (its exact transcript)."""
        if seed is not None:
            self._seed(seed)
        hifi = bool(ref_path and prompt_text)
        flag = "fast_path_hifi" if hifi else "fast_path"
        if ref_path and getattr(self, flag):
            try:
                cache = self._get_prompt_cache(ref_path, ref_key or ref_path, prompt_text if hifi else None)
                clean = re.sub(r"\s+", " ", text.replace("\n", " "))  # same cleanup as VoxCPM.generate()
                wav, _, _ = self.model.tts_model.generate_with_prompt_cache(
                    target_text=clean, prompt_cache=cache, min_len=2, max_len=4096,
                    inference_timesteps=steps, cfg_value=cfg, retry_badcase=retry)
                return wav.squeeze(0).cpu().numpy()
            except Exception as e:  # never let an optimisation break synthesis: fall back to the public API
                setattr(self, flag, False)
                print(f"{'hifi' if hifi else 'clone'} prompt-cache path failed ({e!r}); "
                      f"using model.generate() for that mode from now on", flush=True)
        kw = dict(text=text, cfg_value=cfg, inference_timesteps=steps, retry_badcase=retry)
        if ref_path:
            kw["reference_wav_path"] = ref_path
            if hifi:
                kw.update(prompt_wav_path=ref_path, prompt_text=prompt_text)
        return self.model.generate(**kw)


class FakeEngine:
    """Makes tones whose length follows the text. Lets you test the whole request/response
    pipeline (base64, lesson joining, reference upload, curl scripts) without a GPU."""

    gpu = "fake"
    sr = 48000

    def __init__(self):
        self._seen = set()
        self.ref_encodes = 0
        self.ref_hits = 0
        self.fast_path = True

    def generate(self, text, steps, cfg, seed, retry, ref_path, ref_key=None, prompt_text=None):
        time.sleep(0.05)
        slow = 1.4 if "slow" in text.lower() else 1.0
        dur = min(max(0.06 * len(text) * slow, 0.5), 12.0)
        freq = 220.0
        if ref_path:
            key = ref_key or ref_path
            if key in self._seen:
                self.ref_hits += 1
            else:
                self._seen.add(key)
                self.ref_encodes += 1
            with open(ref_path, "rb") as f:
                freq = 300 + (zlib.crc32(f.read()) % 300)
            if prompt_text:  # hifi: an octave up, so tests can tell the modes apart
                freq *= 2
        elif seed is not None:
            freq = 200 + (seed % 100)
        t = np.arange(int(dur * self.sr)) / self.sr
        return (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def init_engine():
    global ENGINE
    if ENGINE is not None:
        return ENGINE
    t0 = time.perf_counter()
    ENGINE = FakeEngine() if FAKE else RealEngine()
    STATE["load_s"] = round(time.perf_counter() - t0, 1)
    STATE["gpu"] = ENGINE.gpu
    t1 = time.perf_counter()
    try:
        ENGINE.generate(WARM_TEXT, 4, 2.0, None, False, None, None)  # first-call overhead happens here, not on a user's job
    except Exception as e:  # warm-up failure should not kill the worker
        print("warm-up failed:", repr(e), flush=True)
    STATE["warmup_s"] = round(time.perf_counter() - t1, 1)
    STATE["boot_s"] = round(time.perf_counter() - _T0, 1)
    print(f"engine ready: import {getattr(ENGINE, 'import_s', None)}s, weights {getattr(ENGINE, 'weights_s', None)}s "
          f"({getattr(ENGINE, 'source', None)}), warm-up {STATE['warmup_s']}s, boot-to-ready {STATE['boot_s']}s, "
          f"gpu {STATE['gpu']}, dtype {DTYPE}", flush=True)
    return ENGINE


def worker_info():
    return {
        "gpu": STATE["gpu"], "dtype": DTYPE, "optimize": OPTIMIZE, "fake": FAKE,
        "model_load_s": STATE["load_s"], "warmup_s": STATE["warmup_s"], "jobs_served": STATE["jobs"],
        "gpu_arch": getattr(ENGINE, "arch", None), **getattr(ENGINE, "versions", {}),
        "model_source": getattr(ENGINE, "source", None), "import_s": getattr(ENGINE, "import_s", None),
        "weights_s": getattr(ENGINE, "weights_s", None), "boot_to_ready_s": STATE.get("boot_s"),
        "ref_encodes": getattr(ENGINE, "ref_encodes", None), "ref_cache_hits": getattr(ENGINE, "ref_hits", None),
        "prompt_cache_path": getattr(ENGINE, "fast_path", None),
        "prompt_cache_path_hifi": getattr(ENGINE, "fast_path_hifi", None),
    }


def diagnostics():
    """Shown only in ping responses: why a model source was or wasn't picked."""
    def ls(p, n=12):
        try:
            return sorted(os.listdir(p))[:n]
        except Exception:
            return None
    try:
        snap = resolve_cached_snapshot(HF_ID)
    except Exception as e:
        snap = f"error: {e}"
    return {
        "source_setting": MODEL_SOURCE, "hf_id": HF_ID, "cache_root": CACHE_ROOT,
        "cache_root_exists": os.path.isdir(CACHE_ROOT), "cache_entries": ls(CACHE_ROOT),
        "runpod_volume_entries": ls("/runpod-volume"), "resolved_snapshot": snap,
        "baked_exists": os.path.exists(os.path.join(MODEL_DIR, "config.json")),
    }


# --------------------------------------------------------------------------- helpers
def clamp(v, lo, hi):
    try:
        v = float(v)
    except (TypeError, ValueError):
        raise UserError(f"Expected a number between {lo} and {hi}, got {v!r}")
    return max(lo, min(hi, v))


def sniff_ext(raw: bytes) -> str:
    if raw[:4] == b"RIFF": return ".wav"
    if raw[:4] == b"fLaC": return ".flac"
    if raw[:4] == b"OggS": return ".ogg"
    if raw[:3] == b"ID3" or raw[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"): return ".mp3"
    return ".wav"


def write_reference(b64: str):
    """Returns (temp file path, content hash). The hash lets the engine reuse the encoded reference."""
    try:
        raw = base64.b64decode(b64)
    except Exception:
        raise UserError("reference_b64 is not valid base64")
    if not raw:
        raise UserError("reference_b64 is empty")
    if len(raw) > MAX_REF_BYTES:
        raise UserError(f"reference clip too large ({len(raw)} bytes, max {MAX_REF_BYTES})")
    fd, path = tempfile.mkstemp(suffix=sniff_ext(raw))
    with os.fdopen(fd, "wb") as f:
        f.write(raw)
    return path, hashlib.sha1(raw).hexdigest()


def fetch_reference(url: str, ref_id: str, version: str):
    """Download a reference clip from a presigned URL. Returns (path, cache key, was_cached).
    Files stay in REF_DIR and are reused while (reference_id, reference_version) is unchanged."""
    import urllib.parse, urllib.request
    u = urllib.parse.urlparse(url)
    if u.scheme != "https" and not (REF_ALLOW_HTTP and u.scheme == "http"):
        raise UserError("reference_url must be https")
    host = (u.hostname or "").lower()
    if REF_HOSTS and not any(host == h.lstrip(".") or host.endswith(h if h.startswith(".") else "." + h) for h in REF_HOSTS):
        raise UserError("reference_url host is not allowed")
    safe = re.sub(r"[^\w\-]+", "_", ref_id or "ref", flags=re.UNICODE)[:60] or "ref"
    ident = hashlib.sha1(f"{ref_id}|{version}".encode("utf-8")).hexdigest()[:16] if (ref_id or version) else hashlib.sha1(url.split("?")[0].encode()).hexdigest()[:16]
    os.makedirs(REF_DIR, exist_ok=True)
    path = os.path.join(REF_DIR, f"{safe}-{ident}.wav")
    key = f"url:{ident}"
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path, key, True
    try:
        with urllib.request.urlopen(url, timeout=REF_DOWNLOAD_TIMEOUT_S) as r:
            if r.status != 200:
                raise UserError(f"reference download failed (HTTP {r.status})")
            raw = r.read(MAX_REF_BYTES + 1)
    except UserError:
        raise
    except Exception as e:
        raise UserError(f"reference download failed: {type(e).__name__}")
    if not raw:
        raise UserError("reference download was empty")
    if len(raw) > MAX_REF_BYTES:
        raise UserError(f"reference clip too large (max {MAX_REF_BYTES} bytes)")
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        f.write(raw)
    os.replace(tmp, path)
    files = sorted((os.path.join(REF_DIR, n) for n in os.listdir(REF_DIR) if n.endswith(".wav")), key=os.path.getmtime)
    for old in files[:-20]:
        try: os.remove(old)
        except OSError: pass
    return path, key, False


def split_script(script: str):
    parts = re.split(r"(?<=[။.!?])\s*", script.strip())
    return [p.strip() for p in parts if p and p.strip()]


def encode(wav: np.ndarray, sr: int, out_sr):
    """float32 mono -> (base64 16-bit WAV, sample_rate, duration_s)."""
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if out_sr and int(out_sr) != sr:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=int(out_sr))
        sr = int(out_sr)
    wav = np.clip(wav, -1.0, 1.0)
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode("ascii"), sr, round(len(wav) / sr, 3)


def parse_params(inp):
    style = str(inp.get("style") or "").strip().strip("() ")
    seed = inp.get("seed")
    if seed is None and DEFAULT_SEED:
        seed = DEFAULT_SEED
    if seed is not None:
        try:
            seed = int(seed)
        except (TypeError, ValueError):
            raise UserError("seed must be an integer")
    out_sr = inp.get("out_sample_rate")
    if out_sr is not None:
        out_sr = int(clamp(out_sr, 8000, 48000))
    mode = str(inp.get("mode") or "").strip().lower() or None
    if mode is not None and mode not in MODES:
        raise UserError(f"mode must be one of {', '.join(MODES)} (got {inp.get('mode')!r})")
    prompt_text = inp.get("prompt_text") or inp.get("reference_text")
    if prompt_text is not None:
        prompt_text = re.sub(r"\s+", " ", str(prompt_text)).strip() or None
        if prompt_text and len(prompt_text) > MAX_PROMPT_TEXT_CHARS:
            raise UserError(f"prompt_text is {len(prompt_text)} characters; max is {MAX_PROMPT_TEXT_CHARS}")
    return {
        "mode": mode,
        "prompt_text": prompt_text,
        "style": style,
        "steps": int(clamp(inp.get("steps", 10), 2, 30)),
        "cfg": float(clamp(inp.get("cfg", DEFAULT_CFG), 0.5, 5.0)),
        "seed": seed,
        "retry": bool(inp.get("retry_badcase", False)),
        "gap": float(clamp(inp.get("gap_s", 0.7), 0.0, 3.0)),
        "out_sr": out_sr,
        "return_clips": bool(inp.get("return_clips", True)),
    }


def check_sentence(s, where="text"):
    if not isinstance(s, str) or not s.strip():
        raise UserError(f"{where} must be a non-empty string")
    if len(s) > MAX_CHARS:
        raise UserError(f"{where} is {len(s)} characters; max is {MAX_CHARS}. Split it into shorter sentences.")
    return s.strip()


def resolve_mode(p, has_ref, n_sentences):
    """Pick the generation mode before any reference is downloaded. Returns (mode, use_ref, warnings).
    Without an explicit mode the behaviour is the original one: clone with a reference, else design."""
    warnings = []
    mode = p["mode"] or ("clone" if has_ref else "design")
    if mode in ("clone", "hifi") and not has_ref:
        raise UserError(f"mode '{mode}' needs a reference voice (reference_url or reference_b64)")
    if mode == "hifi" and not p["prompt_text"]:
        raise UserError("mode 'hifi' needs prompt_text: the exact transcript of the reference clip")
    if mode == "design" and has_ref:
        warnings.append("reference ignored in design mode")
    if mode != "hifi" and p["prompt_text"]:
        warnings.append(f"prompt_text ignored in {mode} mode (only hifi uses it)")
    if mode == "hifi" and p["style"]:
        warnings.append("style ignored in hifi mode")
    if mode == "design" and n_sentences > 1 and p["seed"] is None:
        warnings.append("design mode without a seed: each sentence may get a different voice; send a seed")
    return mode, mode in ("clone", "hifi"), warnings


def synth(text, p, ref_path, ref_key=None, mode="clone"):
    style = p["style"] if mode != "hifi" else ""   # VoxCPM ignores control instructions in hifi
    t = f"({style}){text}" if style else text
    prompt_text = p["prompt_text"] if mode == "hifi" else None
    t0 = time.perf_counter()
    wav = ENGINE.generate(t, p["steps"], p["cfg"], p["seed"], p["retry"], ref_path, ref_key, prompt_text)
    return np.asarray(wav, dtype=np.float32).reshape(-1), time.perf_counter() - t0


# --------------------------------------------------------------------------- sieng web app mode
OUT_URL_TTL_S = int(os.environ.get("VOX_OUT_URL_TTL_S", "3600"))
_S3 = {"client": None}


def _r2():
    """boto3 client for R2 (created once). Needs the R2_* environment variables."""
    if _S3["client"] is None:
        env = lambda k: (os.environ.get(k) or "").strip()  # copy-pasted values often carry a trailing space or newline
        acct, key, secret = env("R2_ACCOUNT_ID"), env("R2_ACCESS_KEY_ID"), env("R2_SECRET_ACCESS_KEY")
        endpoint = env("R2_ENDPOINT") or (f"https://{acct}.r2.cloudflarestorage.com" if acct else "")
        if not (endpoint and key and secret and env("R2_BUCKET_NAME")):
            raise UserError("R2 is not configured on the worker: set R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY and R2_BUCKET_NAME")
        import boto3
        from botocore.config import Config
        _S3["client"] = boto3.client("s3", endpoint_url=endpoint, aws_access_key_id=key, aws_secret_access_key=secret,
                                     region_name="auto", config=Config(signature_version="s3v4", s3={"addressing_style": "path"}))
    return _S3["client"]


def _safe_part(v, default):
    v = re.sub(r"[^\w\-]", "_", str(v or ""), flags=re.UNICODE).strip("_")
    return (v or default)[:80]


def _output_key(k):
    """body.output_key: where the app wants the clip in R2 (for example <account>/<project>/audios/<line>.wav).
    Must be a plain relative .wav path; anything odd is ignored and the default outputs/... key is used."""
    if not isinstance(k, str) or not (3 < len(k) <= 400) or not k.endswith(".wav"):
        return None
    if k.startswith("/") or "\\" in k or ".." in k.split("/") or any(ord(c) < 32 for c in k) or "//" in k:
        return None
    return k

def _speed_hint(speed):
    try:
        sp = float(speed)
    except (TypeError, ValueError):
        return ""
    if sp < 0.85:
        return "speaking slowly"
    if sp > 1.3:
        return "speaking fast"
    if sp > 1.1:
        return "speaking a little faster"
    return ""


def chunk_for_tts(text, limit=None, max_chunks=None):
    """Split long text into pieces the model can take: sentence by sentence (newlines and ။ . ! ? end a sentence),
    over-long sentences are cut at a comma or space, and if there would be too many pieces, neighbours are merged."""
    limit = limit or MAX_CHARS
    max_chunks = max_chunks or MAX_SENTENCES
    sentences = [x.strip() for ln in re.split(r"[\r\n]+", str(text)) for x in split_script(ln) if x.strip()]
    pieces = []
    for sent in sentences:
        while len(sent) > limit:
            cut = max(sent.rfind(c, 0, limit) for c in ("၊", ",", " ", "\u200b"))
            cut = cut + 1 if cut > limit // 3 else limit
            pieces.append(sent[:cut].strip())
            sent = sent[cut:].strip()
        if sent:
            pieces.append(sent)
    while len(pieces) > max_chunks:  # merge the neighbouring pair with the shortest total
        best = min((i for i in range(len(pieces) - 1) if len(pieces[i]) + len(pieces[i + 1]) + 1 <= limit),
                   key=lambda i: len(pieces[i]) + len(pieces[i + 1]), default=None)
        if best is None:
            raise UserError(f"Text is too long: {sum(map(len, pieces))} characters in {len(pieces)} parts; "
                            f"the limit is {max_chunks} parts of {limit} characters. Send it in smaller pieces.")
        pieces[best:best + 2] = [pieces[best] + " " + pieces[best + 1]]
    return pieces


def _handle_sieng(inp):
    path = str(inp.get("path") or "")
    if str(inp.get("method", "POST")).upper() != "POST" or path.rstrip("/") not in ("/api/generate", "/generate"):
        raise UserError(f"Unsupported route: {inp.get('method')} {path}")
    b = inp.get("body")
    if not isinstance(b, dict) or not str(b.get("text") or "").strip():
        raise UserError("body.text is required")
    style = str(b.get("style") or "").strip()
    hint = _speed_hint(b.get("speed"))
    if hint:
        style = f"{style}, {hint}" if style else hint
    flat = {"text": b["text"], "style": style}
    print(f"sieng request: mode={b.get('mode') or 'auto'} voice={b.get('reference_id')!r} style={style!r} speed={b.get('speed')!r} chars={len(str(b['text']))} seed={b.get('seed', DEFAULT_SEED or None)!r} cfg={b.get('cfg', DEFAULT_CFG)!r}", flush=True)  # shows in the RunPod log
    for k in ("reference_url", "reference_id", "reference_version", "reference_b64", "steps", "cfg", "seed",
              "mode", "prompt_text", "reference_text"):
        if b.get(k) not in (None, ""):
            flat[k] = b[k]
    if b.get("reference_id") and str(b.get("mode") or "").lower() != "design" \
            and not (flat.get("reference_url") or flat.get("reference_b64")):
        raise UserError(f"No reference_url was sent for voice '{b['reference_id']}'")
    chunks = chunk_for_tts(b["text"])
    if not chunks:
        raise UserError("body.text is empty")
    if len(chunks) == 1:
        out = _handle({"input": {**flat, "text": chunks[0]}})
        audio_b64 = out["audio_b64"]
    else:  # long text: generate piece by piece and join with a short pause
        out = _handle({"input": {**flat, "sentences": chunks, "gap_s": 0.35, "return_clips": False}})
        audio_b64 = out["joined_b64"]
    wav_bytes = base64.b64decode(audio_b64)
    process_id = _safe_part(b.get("process_id"), f"p{int(time.time() * 1000)}")
    key = _output_key(b.get("output_key")) or f"outputs/{_safe_part(b.get('group_id'), 'anon')}/{_safe_part(b.get('project_id'), 'none')}/{process_id}.wav"
    s3, bucket = _r2(), os.environ["R2_BUCKET_NAME"].strip()
    try:
        s3.put_object(Bucket=bucket, Key=key, Body=wav_bytes, ContentType="audio/wav")
    except Exception as e:  # botocore ClientError, endpoint errors, ...
        code = getattr(e, "response", {}).get("Error", {}).get("Code") if hasattr(e, "response") else None
        hint = {"Unauthorized": "the R2 key is rejected: it must be an R2 API token's Access Key ID and Secret (not the Cloudflare API token value) with Object Read & Write on this bucket",
                "AccessDenied": "the R2 token has no write permission on this bucket",
                "NoSuchBucket": "R2_BUCKET_NAME does not exist in this account",
                "InvalidAccessKeyId": "R2_ACCESS_KEY_ID is wrong", "SignatureDoesNotMatch": "R2_SECRET_ACCESS_KEY is wrong"}.get(code, "check the R2_* settings and R2_ENDPOINT (EU buckets need the .eu.r2.cloudflarestorage.com endpoint)")
        raise UserError(f"R2 upload failed ({code or type(e).__name__}): {hint}")
    url = s3.generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=OUT_URL_TTL_S)
    # RunPod wraps this dict as {"output": {...}}, which is where the web app reads audio_url and process_id.
    return {"audio_url": url, "process_id": b.get("process_id") or process_id, "key": key,
            "duration_s": out["duration_s"], "gen_s": out["gen_s"], "parts": len(chunks),
            "reference_cached": out.get("reference_cached"), "mode": out.get("mode"),
            "warnings": out.get("warnings", [])}


# --------------------------------------------------------------------------- handler
def _handle(job):
    init_engine()
    inp = job.get("input") or {}
    if not isinstance(inp, dict):
        raise UserError("'input' must be a JSON object")

    if "method" in inp and "path" in inp:
        return _handle_sieng(inp)

    if str(inp.get("action", "")).lower() == "ping":
        return {"ok": True, "worker": worker_info(), "diagnostics": diagnostics()}

    p = parse_params(inp)
    if "sentences" in inp:
        sentences = inp["sentences"]
        if not isinstance(sentences, list) or not sentences:
            raise UserError("'sentences' must be a non-empty list of strings")
        lesson = True
    elif "script" in inp:
        sentences = split_script(str(inp["script"]))
        if not sentences:
            raise UserError("'script' is empty")
        lesson = True
    elif "text" in inp:
        sentences = [inp["text"]]
        lesson = False
    else:
        raise UserError("Provide 'text', 'script' or 'sentences' (or action 'ping').")

    if len(sentences) > MAX_SENTENCES:
        raise UserError(f"{len(sentences)} sentences; max per request is {MAX_SENTENCES}")
    sentences = [check_sentence(s, f"sentence {i + 1}") for i, s in enumerate(sentences)]

    has_ref = bool(inp.get("reference_url") or inp.get("reference_b64"))
    mode, use_ref, warnings = resolve_mode(p, has_ref, len(sentences))

    STATE["jobs"] += 1
    ref_path, ref_key, ref_owned, ref_source, ref_cached = None, None, False, None, None
    if not use_ref:
        pass  # design mode: don't download or decode a reference we won't use
    elif inp.get("reference_url"):
        ref_path, ref_key, ref_cached = fetch_reference(str(inp["reference_url"]), str(inp.get("reference_id") or ""), str(inp.get("reference_version") or ""))
        ref_source = "url"
    elif inp.get("reference_b64"):
        ref_path, ref_key = write_reference(inp["reference_b64"])
        ref_owned, ref_source = True, "b64"
    try:
        sr = ENGINE.sr
        clips, wavs, total_gen = [], [], 0.0
        for i, s in enumerate(sentences, 1):
            wav, secs = synth(s, p, ref_path, ref_key, mode)
            wavs.append(wav)
            total_gen += secs
            b64, out_sr, dur = encode(wav, sr, p["out_sr"])
            clips.append({"index": i, "text": s, "audio_b64": b64, "duration_s": dur, "gen_s": round(secs, 3)})
    finally:
        if ref_owned and ref_path and os.path.exists(ref_path):
            os.remove(ref_path)

    total_audio = sum(len(w) for w in wavs) / sr
    out = {
        "sample_rate": p["out_sr"] or sr,
        "gen_s": round(total_gen, 3),
        "duration_s": round(total_audio, 3),
        "wait_per_audio": round(total_gen / total_audio, 3) if total_audio else None,
        "mode": mode,
        "params": {k: p[k] for k in ("style", "steps", "cfg", "seed", "retry", "out_sr")},
        "used_reference": ref_path is not None,
        "reference_source": ref_source,
        "reference_cached": ref_cached,
        "warnings": warnings,
        "worker": worker_info(),
    }
    if not lesson:
        out["audio_b64"] = clips[0]["audio_b64"]
        return out

    gap = np.zeros(int(sr * p["gap"]), dtype=np.float32)
    joined = []
    for i, w in enumerate(wavs):
        joined.append(w)
        if i < len(wavs) - 1:
            joined.append(gap)
    out["joined_b64"] = encode(np.concatenate(joined), sr, p["out_sr"])[0]
    out["sentence_count"] = len(sentences)
    if p["return_clips"]:
        out["clips"] = clips
    else:
        out["clips"] = [{k: v for k, v in c.items() if k != "audio_b64"} for c in clips]
    return out


def handler(job):
    try:
        return _handle(job)
    except UserError as e:
        return {"error": str(e)}
    except Exception as e:
        traceback.print_exc()
        return {"error": f"{type(e).__name__}: {e}"}


if __name__ == "__main__":
    init_engine()  # load the model before the worker starts taking jobs
    runpod.serverless.start({"handler": handler})