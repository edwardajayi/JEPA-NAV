#!/usr/bin/env python3
"""
Run V-JEPA 2 on a single image or a (T, C, H, W) video clip.
Phase A: sanity check with synthetic/single image.
Phase B: pass frames from Habitat/ReplicaCAD (same API).

Usage:
  python scripts/run_vjepa2_on_frames.py
  python scripts/run_vjepa2_on_frames.py --image path/to/frame.png
  python scripts/run_vjepa2_on_frames.py --frames_dir path/to/frames/

Ref: https://huggingface.co/docs/transformers/model_doc/vjepa2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Add project root for imports if needed
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Hugging Face transformers
from transformers import AutoModel, AutoVideoProcessor

MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"
DEFAULT_FRAMES_PER_CLIP = 64
DEFAULT_CROP_SIZE = 256


def load_model_and_processor(device: str | None = None, dtype: torch.dtype = torch.float16):
    """Load V-JEPA 2 encoder and processor."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoVideoProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(
        MODEL_ID,
        dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        attn_implementation="sdpa",
    )
    if device == "cpu":
        model = model.to(device)
    return processor, model, device


def single_image_to_video_clip(image: np.ndarray, num_frames: int = DEFAULT_FRAMES_PER_CLIP) -> np.ndarray:
    """
    Turn a single image into a video clip by repeating.
    image: (H, W, 3) or (3, H, W), uint8 or float. Will be resized to crop_size x crop_size.
    Returns: (T, C, H, W) in range expected by processor (e.g. 0-255 or 0-1; processor often normalizes).
    """
    if image.ndim == 3 and image.shape[-1] == 3:
        # (H, W, 3) -> (3, H, W)
        image = np.transpose(image, (2, 0, 1))
    assert image.shape[0] == 3, "Expected (3, H, W) or (H, W, 3)"
    # Repeat along new time axis
    clip = np.repeat(image[np.newaxis, ...], num_frames, axis=0)  # (T, 3, H, W)
    return clip


def run_on_clip(
    processor,
    model,
    video: np.ndarray,
    device: str,
    skip_predictor: bool = True,
) -> torch.Tensor:
    """
    video: (T, C, H, W) numpy, T = num_frames, C=3, H,W any (processor will resize/crop).
    Returns: encoder embeddings (batch, seq_len, hidden_size).
    """
    inputs = processor(
        list(video),  # processor may expect list of frames or (T,C,H,W)
        return_tensors="pt",
    )
    # Move to model device
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, skip_predictor=skip_predictor)
    return outputs.last_hidden_state


def main():
    parser = argparse.ArgumentParser(description="Run V-JEPA 2 on image(s) or synthetic clip")
    parser.add_argument("--image", type=str, default=None, help="Path to single image (PNG/JPG)")
    parser.add_argument("--frames_dir", type=str, default=None, help="Path to directory of frames (ordered by name)")
    parser.add_argument("--num_frames", type=int, default=16, help="Frames per clip (repeat single image if one image)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--float32", action="store_true", help="Use float32 instead of float16")
    args = parser.parse_args()

    dtype = torch.float32 if args.float32 else torch.float16
    processor, model, device = load_model_and_processor(device=args.device, dtype=dtype)
    print(f"Model and processor loaded (device={device}).")

    if args.image:
        from PIL import Image
        img = np.array(Image.open(args.image).convert("RGB"))
        video = single_image_to_video_clip(img, num_frames=args.num_frames)
        print(f"Single image repeated to clip shape: {video.shape}")
    elif args.frames_dir:
        from PIL import Image
        frames_dir = Path(args.frames_dir)
        paths = sorted(frames_dir.glob("*.png")) + sorted(frames_dir.glob("*.jpg"))
        if not paths:
            raise FileNotFoundError(f"No PNG/JPG in {frames_dir}")
        target_h, target_w = DEFAULT_CROP_SIZE, DEFAULT_CROP_SIZE
        frames = []
        for p in paths[: args.num_frames]:
            img = np.array(Image.open(p).convert("RGB"))
            if img.ndim == 3 and img.shape[-1] == 3:
                img = np.transpose(img, (2, 0, 1))  # (3,H,W)
            h, w = img.shape[1], img.shape[2]
            if h != target_h or w != target_w:
                pil_img = Image.fromarray(np.transpose(img, (1, 2, 0)))
                pil_img = pil_img.resize((target_w, target_h), Image.BILINEAR)
                img = np.transpose(np.array(pil_img), (2, 0, 1))
            frames.append(img)
        if len(frames) < args.num_frames:
            while len(frames) < args.num_frames:
                frames.append(frames[-1].copy())
        video = np.stack(frames, axis=0)  # (T, 3, H, W)
        print(f"Loaded {len(paths)} frames, clip shape: {video.shape}")
    else:
        # Synthetic: random 256x256 RGB
        video = np.random.randint(0, 255, (args.num_frames, 3, DEFAULT_CROP_SIZE, DEFAULT_CROP_SIZE), dtype=np.uint8)
        print(f"Synthetic clip shape: {video.shape}")

    # Processor expects (T, C, H, W). It will resize/crop to model's crop_size (256).
    inputs = processor(video, return_tensors="pt")
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, skip_predictor=True)

    emb = outputs.last_hidden_state
    print(f"Encoder output shape: {emb.shape}")
    print(f"  -> (batch_size, sequence_length, hidden_size) = {emb.shape}")
    if emb.numel() > 0:
        print(f"  -> mean={emb.float().mean().item():.4f}, std={emb.float().std().item():.4f}")
    return emb


if __name__ == "__main__":
    main()
