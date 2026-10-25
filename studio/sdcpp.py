"""Text-to-image on a CPU runner with stable-diffusion.cpp (MIT) and quantised GGUF weights.

The server keeps the model in memory, so a shard of ten images pays the load cost once. Every model here is
Apache-2.0 (checked on its Hugging Face card); the registry is also what the benchmark compares.
"""
from __future__ import annotations

import base64
import io
import os
import socket
import subprocess
import time
import zipfile
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from dataclasses import dataclass, field
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(os.environ.get("PANGAEA_MODELS", ROOT / "work" / "models"))
SD_TAG = "master-950-f2f168f"
SD_ZIP = "sd-master-f2f168f-bin-Linux-Ubuntu-24.04-x86_64.zip"
HF = "https://huggingface.co/{repo}/resolve/main/{file}"


@dataclass
class Weights:
    role: str  # diffusion-model | llm | vae | lora
    repo: str
    file: str

    @property
    def path(self) -> Path:
        return CACHE / self.repo.replace("/", "__") / Path(self.file).name


@dataclass
class Candidate:
    name: str
    license: str
    weights: list[Weights]
    steps: int
    cfg: float = 1.0
    sampler: str = "euler"
    width: int = 1024
    height: int = 576
    extra: list[str] = field(default_factory=list)
    prompt_suffix: str = ""
    threads: int | None = None


_ZIMAGE_VAE = Weights("vae", "Comfy-Org/z_image_turbo", "split_files/vae/ae.safetensors")
_KLEIN_VAE_SMALL = Weights("vae", "black-forest-labs/FLUX.2-small-decoder", "full_encoder_small_decoder.safetensors")
_KLEIN_VAE_FULL = Weights("vae", "black-forest-labs/FLUX.2-klein-4B", "vae/diffusion_pytorch_model.safetensors")

CANDIDATES = {
    "klein4b-q8": Candidate("klein4b-q8", "Apache-2.0", [
        Weights("diffusion-model", "leejet/FLUX.2-klein-4B-GGUF", "flux-2-klein-4b-Q8_0.gguf"),
        Weights("llm", "unsloth/Qwen3-4B-GGUF", "Qwen3-4B-Q8_0.gguf"), _KLEIN_VAE_FULL], steps=4),
    "klein4b-q8-small-vae": Candidate("klein4b-q8-small-vae", "Apache-2.0", [
        Weights("diffusion-model", "leejet/FLUX.2-klein-4B-GGUF", "flux-2-klein-4b-Q8_0.gguf"),
        Weights("llm", "unsloth/Qwen3-4B-GGUF", "Qwen3-4B-Q8_0.gguf"), _KLEIN_VAE_SMALL], steps=4),
    "klein4b-q4": Candidate("klein4b-q4", "Apache-2.0", [
        Weights("diffusion-model", "leejet/FLUX.2-klein-4B-GGUF", "flux-2-klein-4b-Q4_0.gguf"),
        Weights("llm", "unsloth/Qwen3-4B-GGUF", "Qwen3-4B-Q4_K_M.gguf"), _KLEIN_VAE_FULL], steps=4),
    "klein4b-q8-1344": Candidate("klein4b-q8-1344", "Apache-2.0", [
        Weights("diffusion-model", "leejet/FLUX.2-klein-4B-GGUF", "flux-2-klein-4b-Q8_0.gguf"),
        Weights("llm", "unsloth/Qwen3-4B-GGUF", "Qwen3-4B-Q8_0.gguf"), _KLEIN_VAE_FULL], steps=4,
        width=1344, height=768),
    "klein4b-q4-768": Candidate("klein4b-q4-768", "Apache-2.0", [
        Weights("diffusion-model", "leejet/FLUX.2-klein-4B-GGUF", "flux-2-klein-4b-Q4_0.gguf"),
        Weights("llm", "unsloth/Qwen3-4B-GGUF", "Qwen3-4B-Q4_K_M.gguf"), _KLEIN_VAE_SMALL], steps=4,
        width=768, height=432),
    "klein4b-q4-768-t2": Candidate("klein4b-q4-768-t2", "Apache-2.0", [
        Weights("diffusion-model", "leejet/FLUX.2-klein-4B-GGUF", "flux-2-klein-4b-Q4_0.gguf"),
        Weights("llm", "unsloth/Qwen3-4B-GGUF", "Qwen3-4B-Q4_K_M.gguf"), _KLEIN_VAE_SMALL], steps=4,
        width=768, height=432, threads=2),
    "klein4b-q4-672": Candidate("klein4b-q4-672", "Apache-2.0", [
        Weights("diffusion-model", "leejet/FLUX.2-klein-4B-GGUF", "flux-2-klein-4b-Q4_0.gguf"),
        Weights("llm", "unsloth/Qwen3-4B-GGUF", "Qwen3-4B-Q4_K_M.gguf"), _KLEIN_VAE_SMALL], steps=4,
        width=672, height=384),
    "klein4b-q8-768": Candidate("klein4b-q8-768", "Apache-2.0", [
        Weights("diffusion-model", "leejet/FLUX.2-klein-4B-GGUF", "flux-2-klein-4b-Q8_0.gguf"),
        Weights("llm", "unsloth/Qwen3-4B-GGUF", "Qwen3-4B-Q8_0.gguf"), _KLEIN_VAE_SMALL], steps=4,
        width=768, height=432),
    "zimage-q8": Candidate("zimage-q8", "Apache-2.0", [
        Weights("diffusion-model", "leejet/Z-Image-Turbo-GGUF", "z_image_turbo-Q8_0.gguf"),
        Weights("llm", "unsloth/Qwen3-4B-Instruct-2507-GGUF", "Qwen3-4B-Instruct-2507-Q8_0.gguf"), _ZIMAGE_VAE],
        steps=8),
    "zimage-q4k": Candidate("zimage-q4k", "Apache-2.0", [
        Weights("diffusion-model", "leejet/Z-Image-Turbo-GGUF", "z_image_turbo-Q4_K.gguf"),
        Weights("llm", "unsloth/Qwen3-4B-Instruct-2507-GGUF", "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"), _ZIMAGE_VAE],
        steps=8),
    "zimage-q8-6step": Candidate("zimage-q8-6step", "Apache-2.0", [
        Weights("diffusion-model", "leejet/Z-Image-Turbo-GGUF", "z_image_turbo-Q8_0.gguf"),
        Weights("llm", "unsloth/Qwen3-4B-Instruct-2507-GGUF", "Qwen3-4B-Instruct-2507-Q8_0.gguf"), _ZIMAGE_VAE],
        steps=6),
    "qwen2512-q3-lightning4": Candidate("qwen2512-q3-lightning4", "Apache-2.0", [
        Weights("diffusion-model", "unsloth/Qwen-Image-2512-GGUF", "qwen-image-2512-Q3_K_S.gguf"),
        Weights("llm", "mradermacher/Qwen2.5-VL-7B-Instruct-GGUF", "Qwen2.5-VL-7B-Instruct.Q4_K_M.gguf"),
        Weights("vae", "Comfy-Org/Qwen-Image_ComfyUI", "split_files/vae/qwen_image_vae.safetensors"),
        Weights("lora", "lightx2v/Qwen-Image-2512-Lightning", "Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors")],
        steps=4, extra=["--flow-shift", "3"],
        prompt_suffix="<lora:Qwen-Image-2512-Lightning-4steps-V1.0-bf16:1>"),
}
DEFAULT = os.environ.get("PANGAEA_IMAGE_MODEL", "klein4b-q8-768")


def _download_delay(response, attempt: int) -> float:
    """Respect a provider's rate-limit delay without unbounded runner waits."""
    fallback = min(60, 30 * (attempt + 1))
    if response is None or response.status_code not in (429, 503):
        return fallback
    retry = response.headers.get("Retry-After", "")
    try:
        seconds = float(retry)
    except (TypeError, ValueError):
        try:
            seconds = (parsedate_to_datetime(retry) - datetime.now(timezone.utc)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return fallback
    return max(fallback, min(180, seconds))


def _download(url: str, path: Path, tries: int = 4) -> Path:
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    for attempt in range(tries):
        try:
            with requests.get(url, stream=True, timeout=120, headers={"User-Agent": "pangaea-nights-studio/1.0"}) as r:
                r.raise_for_status()
                with part.open("wb") as fh:
                    for chunk in r.iter_content(1 << 22):
                        fh.write(chunk)
            part.rename(path)
            return path
        except requests.RequestException as exc:
            part.unlink(missing_ok=True)
            response = exc.response
            if response is not None and 400 <= response.status_code < 500 and response.status_code not in (408, 429):
                raise
            if attempt == tries - 1:
                raise
            time.sleep(_download_delay(response, attempt))
    return path


def binary() -> Path:
    folder = CACHE / f"sdcpp-{SD_TAG}"
    exe = folder / "sd-server"
    if not exe.exists():
        z = _download(f"https://github.com/leejet/stable-diffusion.cpp/releases/download/{SD_TAG}/{SD_ZIP}",
                      CACHE / SD_ZIP)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(folder)
        z.unlink()
        found = next(folder.rglob("sd-server"), None)
        if found is None:
            raise RuntimeError("stable-diffusion.cpp release has no sd-server")
        for f in found.parent.iterdir():
            f.chmod(0o755)
        exe = found
    return next(folder.rglob("sd-server"))


def fetch(candidate: Candidate) -> None:
    for w in candidate.weights:
        _download(HF.format(repo=w.repo, file=w.file), w.path)


def _port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Server:
    """with Server(candidate) as s: png_bytes = s.generate(prompt, seed)"""

    def __init__(self, candidate: Candidate | str = DEFAULT, threads: int | None = None):
        self.c = CANDIDATES[candidate] if isinstance(candidate, str) else candidate
        self.threads = threads or self.c.threads or os.cpu_count() or 4
        self.port = _port()
        self.proc: subprocess.Popen | None = None
        self.load_seconds = 0.0

    def __enter__(self) -> "Server":
        fetch(self.c)
        exe = binary()
        args = [str(exe), "--listen-ip", "127.0.0.1", "--listen-port", str(self.port), "-t", str(self.threads),
                "--diffusion-fa", "--cfg-scale", str(self.c.cfg)]
        for w in self.c.weights:
            if w.role == "lora":
                args += ["--lora-model-dir", str(w.path.parent)]
            else:
                args += [f"--{w.role}", str(w.path)]
        args += self.c.extra
        env = dict(os.environ, LD_LIBRARY_PATH=str(exe.parent))
        self.log = (CACHE / f"sd-server-{self.port}.log").open("wb")
        start = time.monotonic()
        self.proc = subprocess.Popen(args, stdout=self.log, stderr=subprocess.STDOUT, env=env)
        while time.monotonic() - start < 900:
            if self.proc.poll() is not None:
                raise RuntimeError("sd-server exited: " + self.tail())
            try:
                if requests.get(f"http://127.0.0.1:{self.port}/sdapi/v1/sd-models", timeout=3).ok:
                    self.load_seconds = time.monotonic() - start
                    return self
            except requests.RequestException:
                pass
            time.sleep(2)
        raise RuntimeError("sd-server did not become ready")

    def tail(self, n: int = 1500) -> str:
        self.log.flush()
        return Path(self.log.name).read_bytes()[-n:].decode(errors="replace")

    def peak_rss_mb(self) -> float:
        try:
            for line in Path(f"/proc/{self.proc.pid}/status").read_text().splitlines():
                if line.startswith("VmHWM"):
                    return int(line.split()[1]) / 1024
        except OSError:
            pass
        return 0.0

    def __exit__(self, *exc) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.log.close()

    def generate(self, prompt: str, seed: int, width: int | None = None, height: int | None = None) -> bytes:
        body = {"prompt": prompt + self.c.prompt_suffix, "width": width or self.c.width,
                "height": height or self.c.height, "steps": self.c.steps, "cfg_scale": self.c.cfg, "seed": seed,
                "batch_size": 1, "sampler_name": self.c.sampler}
        r = requests.post(f"http://127.0.0.1:{self.port}/sdapi/v1/txt2img", json=body, timeout=3600)
        if not r.ok:
            raise RuntimeError(f"txt2img failed ({r.status_code}): {r.text[:300]} | {self.tail(600)}")
        images = r.json().get("images") or []
        if not images:
            raise RuntimeError("txt2img returned no image: " + self.tail(600))
        return base64.b64decode(images[0].split(",", 1)[-1])


def to_pil(data: bytes):
    from PIL import Image

    return Image.open(io.BytesIO(data)).convert("RGB")
