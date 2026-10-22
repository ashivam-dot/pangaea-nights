"""Automatic image review: reject people, text, collages, glitches, blank or blown-out frames and off-prompt scenes.

CLIP ViT-B/32 (OpenAI weights, MIT) gives zero-shot labels and prompt similarity; the LAION aesthetic head for the
same model (MIT) gives a 1-10 aesthetic score. Simple pixel statistics catch blank and broken frames.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

AESTHETIC_URL = "https://github.com/LAION-AI/aesthetic-predictor/raw/main/sa_0_4_vit_b_32_linear.pth"
LABELS = {
    "person": ["a photo of a person", "a human face", "a human hand", "a crowd of people", "a diver or a swimmer"],
    "text": ["text, letters, words or a watermark", "a sign with writing", "a page of a book"],
    "broken": ["a blurry, distorted, glitched image", "a collage or grid of several pictures",
               "a cartoon or comic illustration", "a screenshot of a user interface"],
    "good": ["a calm nature documentary scene", "an underwater scene", "a prehistoric animal in its habitat",
             "a quiet landscape at night", "a close-up of a natural texture", "a misty ancient forest",
             "a seascape under a dark sky"],
}
LIMITS = {"person": 0.30, "text": 0.30, "broken": 0.40, "similarity": 0.19, "aesthetic": 4.6,
          "dark": 10.0, "bright": 215.0, "flat": 9.0, "sharp": 8.0}

_model = None


@dataclass
class Verdict:
    ok: bool
    reasons: list[str]
    person: float
    text: float
    broken: float
    similarity: float
    aesthetic: float
    brightness: float
    contrast: float
    sharpness: float

    def score(self) -> float:
        """Higher is better; used to pick the best of several takes."""
        return self.aesthetic + 12 * self.similarity - 4 * (self.person + self.text + self.broken)

    def as_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def _load():
    global _model
    if _model is None:
        import open_clip
        import torch

        model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        model.eval()
        tok = open_clip.get_tokenizer("ViT-B-32")
        prompts = [p for group in LABELS.values() for p in group]
        with torch.no_grad():
            label_emb = model.encode_text(tok(prompts)).float()
            label_emb /= label_emb.norm(dim=-1, keepdim=True)
        head = torch.nn.Linear(512, 1)
        path = Path(torch.hub.get_dir()) / "sa_0_4_vit_b_32_linear.pth"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.hub.download_url_to_file(AESTHETIC_URL, str(path), progress=False)
        head.load_state_dict(torch.load(path, map_location="cpu"))
        head.eval()
        _model = (model, preprocess, tok, label_emb, head, prompts)
    return _model


def pixel_stats(img: Image.Image) -> tuple[float, float, float]:
    """Mean luminance, luminance std, and Laplacian sharpness (std of the 3x3 Laplacian) on a 512 px copy."""
    g = np.asarray(img.convert("L").resize((512, round(512 * img.height / img.width))), dtype=np.float32)
    lap = (-4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:])
    return float(g.mean()), float(g.std()), float(lap.std())


def review(img: Image.Image, prompt: str) -> Verdict:
    import torch

    model, preprocess, tok, label_emb, head, prompts = _load()
    with torch.no_grad():
        emb = model.encode_image(preprocess(img).unsqueeze(0)).float()
        emb /= emb.norm(dim=-1, keepdim=True)
        pemb = model.encode_text(tok([prompt[:300]])).float()
        pemb /= pemb.norm(dim=-1, keepdim=True)
        probs = (100 * emb @ label_emb.T).softmax(dim=-1)[0].numpy()
        sim = float((emb @ pemb.T)[0, 0])
        aesthetic = float(head(emb)[0, 0])
    groups, i = {}, 0
    for name, group in LABELS.items():
        groups[name] = float(probs[i:i + len(group)].sum())
        i += len(group)
    bright, contrast, sharp = pixel_stats(img)
    reasons = []
    for key in ("person", "text", "broken"):
        if groups[key] > LIMITS[key]:
            reasons.append(f"{key} {groups[key]:.2f}")
    if sim < LIMITS["similarity"]:
        reasons.append(f"off-prompt {sim:.3f}")
    if aesthetic < LIMITS["aesthetic"]:
        reasons.append(f"aesthetic {aesthetic:.2f}")
    if bright < LIMITS["dark"]:
        reasons.append(f"too dark {bright:.0f}")
    if bright > LIMITS["bright"]:
        reasons.append(f"too bright {bright:.0f}")
    if contrast < LIMITS["flat"]:
        reasons.append(f"flat {contrast:.1f}")
    if sharp < LIMITS["sharp"]:
        reasons.append(f"mushy {sharp:.1f}")
    return Verdict(not reasons, reasons, groups["person"], groups["text"], groups["broken"], sim, aesthetic,
                   bright, contrast, sharp)


def embed(img: Image.Image) -> np.ndarray:
    """The unit CLIP image embedding (512 floats), for image-to-image similarity."""
    import torch

    model, preprocess = _load()[:2]
    with torch.no_grad():
        emb = model.encode_image(preprocess(img.convert("RGB")).unsqueeze(0)).float()
        emb /= emb.norm(dim=-1, keepdim=True)
    return emb[0].numpy().astype(np.float32)


# Figure check for the bright, dramatic ladder scenes. The softmax over LABELS has calm sleep-documentary "good"
# labels only, so a sunlit fireball, smoke column or root tunnel matches none of them and its mass leaks into "person"
# (0.30-0.72 on empty scenes in dry run 2), while small human silhouettes in a landscape score as low as 0.00. This
# check is a two-sided contrast: deep-time scene types against people. On 31 empty ladder takes it scored at most 0.38;
# on 14 Commons photos of people (portraits, hikers, silhouettes, beach figures) at least 0.64. The face and hand labels
# catch a stray body part: hands/faces pasted into 18 empty takes were caught 16/18 with them, 13/18 without.
FIGURE_NO = ["an empty landscape", "a prehistoric animal", "a dramatic sky with fire, smoke or lightning",
             "a cave, burrow or hollow tree", "the sea, a shore or underwater", "plants, a forest or roots",
             "a volcano or lava"]
FIGURE_YES = ["a landscape with a person in it", "a human silhouette", "a person", "people",
              "a small figure of a person in the distance", "a diver or a swimmer", "a human face",
              "a human hand or arm"]
FIGURE_LIMIT = 0.50
_figure_emb = None


def figure(img: Image.Image) -> float:
    """Probability mass on the people side of the FIGURE contrast (0-1)."""
    global _figure_emb
    import torch

    model, _, tok = _load()[:3]
    with torch.no_grad():
        if _figure_emb is None:
            _figure_emb = model.encode_text(tok(FIGURE_NO + FIGURE_YES)).float()
            _figure_emb /= _figure_emb.norm(dim=-1, keepdim=True)
        emb = torch.from_numpy(embed(img)).unsqueeze(0)
        probs = (100 * emb @ _figure_emb.T).softmax(dim=-1)[0]
    return float(probs[len(FIGURE_NO):].sum())


def pack(vec: np.ndarray) -> str:
    """An embedding as base64 float16, small enough to keep in JSON and in the ledger."""
    import base64

    return base64.b64encode(np.asarray(vec, np.float16).tobytes()).decode()


def unpack(text: str) -> np.ndarray:
    import base64

    v = np.frombuffer(base64.b64decode(text), np.float16).astype(np.float32)
    return v / max(float(np.linalg.norm(v)), 1e-9)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-9))
