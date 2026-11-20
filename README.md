My Runpod

Nothing is compiled on a paid GPU: flash-attn wheels are built on free GitHub runners,
and model weights come from RunPod's cached models, not the image.

```
.github/workflows/build-wheels.yml   compile flash-attn per GPU arch  -> GitHub Releases
.github/workflows/build-image.yml    build slim image per GPU arch    -> ghcr.io/<owner>/<repo>:smXX
Dockerfile                           one venv: nanovllm + VoxCPM + flash-attn wheel + runpod
handler.py                           RunPod handler (modes, voices, latent cache, smoke test)
client.py                            call the endpoint from your machine
versions.env                         pins shared by both workflows
voices/                              preset voices baked into the image (Grey.wav + Grey.txt)
```

## 1. One-time setup

1. Create a **public** GitHub repository (free, unlimited runner minutes, 4-core runners)
   and push these files.
2. Add your preset voice: `voices/Grey.wav` and its exact transcript `voices/Grey.txt`.
3. Check `versions.env`. `NANOVLLM_REF` is the commit you benchmarked; a full SHA is safer.

## 2. Build the flash-attn wheels (rarely)

Actions > **Build flash-attn wheels** > Run workflow. Pick the architectures you'll serve:

| Arch | GPUs |
|---|---|
| `80` | Ampere / Ada: RTX 3090, RTX 4090, A4000, A5000, A6000, A100, L4, L40S |
| `90` | Hopper: H100, H200 |
| `100` | Blackwell datacenter: B200 |
| `120` | Blackwell workstation / consumer: RTX PRO 4000/6000 Blackwell, RTX 5090 |

Each arch is a separate job (about 1-3 h on a free runner) and becomes a release named
`fa<version>-nv<ref>-sm<arch>`. Rerun only when `versions.env` changes.

## 3. Build the image (every code change)

Actions > **Build and push image** (also runs automatically on pushes to `main` that touch
`handler.py`, `Dockerfile`, `voices/` or `versions.env`). Takes ~10-20 min per arch and pushes
`ghcr.io/<owner>/<repo>:sm120` plus an immutable `:sm120-<sha>` tag.

New GHCR packages are **private**. Either make the package public (GitHub > your profile >
Packages > package > Package settings > Change visibility) or add GHCR credentials in RunPod
(Settings > Container Registry Auth, using a GitHub token with `read:packages`).

## 4. Create the RunPod endpoint

Serverless > New Endpoint > Import from Docker Registry:

| Setting | Value |
|---|---|
| Container image | `ghcr.io/<owner>/<repo>:sm120` (pin `:sm120-<sha>` for production) |
| GPU types | Only GPUs matching the tag (see table above) |
| CUDA version (advanced) | 13.x only: the image's torch is built for CUDA 13.2 |
| **Model** | `openbmb/VoxCPM2` (cached model: no download, no download billing) |
| Container disk | ~20 GB |
| Active workers | 0 (raise to 1 only for steady traffic) |
| Max workers | to taste; caps spend |
| Idle timeout | 5-30 s (short for rare traffic, longer for bursts) |
| Execution timeout | 600 s |
| FlashBoot | On |
| Environment variables | `ENGINE=nano` (or `default`), optional ones below |

The handler reads the cached model from `/runpod-volume/huggingface-cache/hub`. If the Model
field is not set, it falls back to downloading, which is slow and billed.

## 5. Test

```bash
export RUNPOD_API_KEY=... RUNPOD_ENDPOINT_ID=...
python client.py --warmup                                    # loads a worker, lists voices
python client.py --text "မင်္ဂလာပါ။ ဒီနေ့ ရာသီဥတု အရမ်းသာယာပါတယ်။" \
                 --instruction "A young woman, gentle and warm voice" -o design.wav
python client.py --text-file story.txt --voice Grey -o clone.wav
python client.py --text-file story.txt --voice Grey --mode hifi -o hifi.wav
```

Check the worker log for `flash-attn smoke test OK` on the first start.

## API

`POST /runsync` (or `/run` + `/status/{id}`) with `{"input": {...}}`:

| Field | Default | Meaning |
|---|---|---|
| `text` | required | Text to speak (any of VoxCPM2's 30 languages, Burmese included) |
| `mode` | `clone` if a reference is given, else `design` | `design`, `clone` or `hifi` |
| `instruction` | none | Voice description (design) or style (clone), e.g. `"slightly faster, cheerful tone"`. Ignored in hifi |
| `voice` | none | Preset from `voices/` (its `.txt` becomes `prompt_text`) |
| `reference_audio_base64` | none | Your own reference clip, base64 |
| `reference_audio_format` | `wav` | Format of the base64 clip |
| `reference_audio_url` | none | Reference clip the worker downloads (disable with `ALLOW_REF_URL=0`) |
| `prompt_text` | from voice | Exact transcript of the reference (hifi) |
| `cfg_value` | 2.0 | Guidance, typically 1.0-3.0 (lower if long output gets noisy) |
| `inference_timesteps` | 10 | Diffusion steps, 4-30; more is slower and more detailed |
| `output_format` | `wav` | `wav` or `flac` (16-bit; flac is ~half the size) |
| `split_long_text` | true | Split long text at sentence ends (`။ . ! ?`) for stability. Not applied in design mode, where each segment would get a new voice |
| `warmup` | false | Only load the engine; returns load time and voices |

Response:

```json
{"audio_base64": "...", "format": "wav", "sample_rate": 48000, "duration_s": 12.4,
 "generation_s": 1.6, "rtf": 0.129, "engine": "nano", "mode": "clone", "segments": 1,
 "cold_start_load_s": 0.0, "reference_cache_hit": true, "warnings": []}
```

Errors come back as `{"error": "..."}`. Bad input leaves the worker running; engine failures
also set `refresh_worker` so RunPod replaces the worker.

Responses carry the audio inline as base64. For long audio prefer `flac`; for very long audio,
upload to object storage from the handler and return a URL instead (RunPod limits payload sizes).

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `ENGINE` | `nano` | `nano` or `default` |
| `MODEL_ID` | `openbmb/VoxCPM2` | Must match the endpoint's Model field |
| `MAX_CONCURRENCY` | 8 | Jobs one nano worker takes at once (default engine: always 1) |
| `MAX_NUM_SEQS` | 16 | nano scheduler slots (segments of concurrent jobs share them) |
| `GPU_MEM_UTIL` | 0.85 | Fraction of VRAM nano may use |
| `MAX_TEXT_CHARS` | 3000 | Reject longer text |
| `MAX_SEGMENT_CHARS` | 250 | Split size for long text |
| `LATENT_CACHE_SIZE` | 32 | Reference clips whose latents stay cached per worker (nano) |
| `RETRY_BADCASE` | 1 | VoxCPM's automatic retry of bad outputs (default engine) |
| `SMOKE_TEST` | 1 | Check flash-attn and the GPU arch at startup (nano) |
| `REFRESH_ON_ERROR` | 1 | Replace the worker after an engine error |
| `ALLOW_REF_URL` | 1 | Allow `reference_audio_url` |
| `SAMPLE_RATE` | auto | Override output sample rate detection |

## Cost notes

- Billed time per cold start = engine load (~`cold_start_load_s` on the first request) plus
  requests plus the idle timeout. FlashBoot and a sensible idle timeout matter most.
- One nano worker serves up to `MAX_CONCURRENCY` requests at once; that, not single-request
  RTF, is where most of its cost advantage comes from under real traffic.
- Reusing a voice is cheaper on nano: its latents are cached per worker
  (`reference_cache_hit: true`).
- The engine loads on a worker's first job. Send `{"warmup": true}` after deploying or before
  expected traffic so a user request doesn't pay for it.

## Local test (on any GPU pod)

```bash
docker run --gpus all -e ENGINE=nano -v $PWD/test_input.json:/app/test_input.json \
  ghcr.io/<owner>/<repo>:sm120
```

The RunPod SDK runs `test_input.json` once and exits. The first run downloads the model
(no RunPod cache outside Serverless).
