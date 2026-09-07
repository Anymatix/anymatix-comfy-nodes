# anymatix-comfy-nodes

The ComfyUI nodes [Anymatix](https://www.anymatix.com) runs on.

Anymatix is a desktop application for generative image and video work. It drives
a ComfyUI it manages itself, and the workflows it ships are addressed **by URL**
rather than by filename: a card says *this* checkpoint, *this* LoRA, *this* VAE,
and the node downloads it, verifies it, deduplicates it and hands the loader a
path. This pack is that mechanism, plus the save and utility nodes those
workflows need.

It is published so that a workflow exported from Anymatix opens in a stock
ComfyUI. You do not need Anymatix to use it.

```
comfy node install anymatix-comfy-nodes
```

## What it gives you

**URL-addressed loaders.** `AnymatixFetcher` takes a URL and a model type and
returns the local path of the file, downloading it if it is not there. It
resumes interrupted transfers, verifies size, computes a SHA-256 and reuses a
model it already holds under another URL. Around it sit thin twins of ComfyUI's
own loaders — checkpoint, LoRA, VAE, CLIP (single/dual/triple/quadruple), UNET,
ControlNet, CLIP-Vision, upscale, model patch, audio encoder, LTX audio VAE and
text encoder, latent upscale, SAM2 — which differ from the originals in one
respect: they take a path instead of a dropdown, because the file may not have
existed when the graph was written.

**Save nodes.** `AnymatixImageSave`, `AnymatixSaveAnimatedMP4`,
`AnymatixSaveAudio`, `AnymatixSaveJson` — the results Anymatix reads back.
`AnymatixImageToVideo` wraps a batch of frames plus optional audio as a VIDEO.

**Utilities.** `AnymatixCLIPSeg` (text-prompted segmentation),
`AnymatixMaskImage`, `AnymatixMaskToSAMcoord` (a mask becomes SAM point
prompts), `AnymatixAudioDuration` and `AnymatixFrameCount` (read a length off
the file instead of asking for it), `AnymatixLTXResizeToClosestValidSize`.

**Offline preprocessors.** `AnymatixDWPreprocessor`, `AnymatixHEDPreprocessor`
and `AnymatixZoeDepthAnythingPreprocessor` wrap the `comfyui_controlnet_aux`
implementations but load their weights from a local path, so they work with
`HF_HUB_OFFLINE=1` and never call the Hub at run time.

## Optional siblings

Some nodes wrap other packs. Each is checked when it is used, not at import, so
a missing one costs you that node and nothing else:

| node | needs |
|---|---|
| `AnymatixUNETLoaderGGUF` | [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) |
| `AnymatixSAM2Loader` | [ComfyUI-segment-anything-2](https://github.com/kijai/ComfyUI-segment-anything-2) |
| `AnymatixDWPreprocessor`, `AnymatixHEDPreprocessor` | [comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux) |
| `AnymatixChatterboxPackFromFetchedName` | ComfyUI-Chatterbox |

## HTTP routes this pack adds — read this before exposing ComfyUI

Installing this pack registers routes on **your** ComfyUI server. If you run
ComfyUI with `--listen`, they are reachable by anything that can reach it, with
no authentication, exactly like ComfyUI's own routes.

| route | what it does |
|---|---|
| `GET /anymatix/log`, `/anymatix/resources`, `/anymatix/cache_size`, `/anymatix/storage_location`, `/anymatix/host_compute_metrics` | read-only: logs, the model listing, cache size, CPU/GPU load |
| `GET /anymatix/{output,input,models}/…` | serves files from those directories |
| `POST /anymatix/uploadAsset` | writes a file named `<sha256>.<ext>` into the input dir or a model root; the hash is verified after the write and the name is validated before it |
| `POST /anymatix/expunge`, `/anymatix/release_results`, `/anymatix/cache_gc` | **delete** cached results and input assets |
| `POST /anymatix/delete_resource` | **deletes a model file and its sidecar**, by URL; restricted to ComfyUI's own model roots |
| `GET /anymatix/reboot` | clears the queue and unloads models (`?deep=true` also reloads modules) |
| `POST /anymatix/heartbeat` | see below |

**The heartbeat cannot stop your machine.** Anymatix uses it as a dead-man's
switch: if the app stops pinging, the ComfyUI it started exits, and on a rented
cloud GPU the pod is stopped so the billing ends. That only ever arms when
`ANYMATIX_HEARTBEAT_PORT` is set, which Anymatix's own launcher does and nothing
else does. On your ComfyUI the route answers `{"armed": false}` and arms
nothing.

## Cache expiry

Results under `output/anymatix/results` and hash-named files in `input/` are
swept on a timer, since both are recomputable. Every interval is an environment
variable and `0` disables that half:

| variable | default |
|---|---|
| `ANYMATIX_CACHE_GC_INTERVAL` | 600 s (`0` disables the sweep entirely) |
| `ANYMATIX_CACHE_TTL_RESULTS` | 86400 s |
| `ANYMATIX_CACHE_TTL_INPUTS` | 3600 s |
| `ANYMATIX_CACHE_MIN_AGE` | 900 s |

Nothing referenced by the queue is ever swept, and serving a result renews it.

## Downloads

Large files are fetched in parallel byte ranges when the server supports
`Range`, and fall back to one stream when it does not — which is also the resume
path. Bytes accumulate in `<name>.part` and the file is given its real name only
when its size matches what the server declared, so a partial download can never
be mistaken for a model. `aiohttp` and `aiofiles` make the parallel path
available; without them the single stream is used.

We publish no speed numbers here. An earlier version of this file did, and
nothing in the repository measured them.

## Requirements

ComfyUI, and the packages in `pyproject.toml`. Everything else this code
imports — torch, requests, transformers, spandrel — is already a ComfyUI
dependency.

## Third-party code

* `poly_decomp.py` — the Bayazit convex decomposition, ported from
  [wsilva32/poly_decomp.py](https://github.com/wsilva32/poly_decomp.py);
  the mask-to-SAM approach follows
  [Glidias/mask2sam](https://github.com/Glidias/mask2sam).
* `AnymatixSAM2Loader` is adapted from `DownloadAndLoadSAM2Model` in
  [kijai/ComfyUI-segment-anything-2](https://github.com/kijai/ComfyUI-segment-anything-2)
  (Apache 2.0).
* `AnymatixMaskImage` reproduces matplotlib's `Greys_r` ramp from its own
  control points rather than importing matplotlib.

## Licence

GNU General Public License v3.0 — see [LICENSE](LICENSE).
