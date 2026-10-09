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
          "dark": 10.0, "bright": 215.0, "flat": 9.0, "sharp": 12.0}

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
