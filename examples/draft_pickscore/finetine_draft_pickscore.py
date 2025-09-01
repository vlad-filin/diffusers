
import os
import math
import csv
import time
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDIMScheduler,
)
from diffusers.models.attention_processor import LoRAAttnProcessor2_0
from diffusers.utils import AttnProcsLayers
from transformers import AutoTokenizer, CLIPTextModel, AutoModel, AutoConfig

from PIL import Image



def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def now_utc():
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())



class PromptDataset(Dataset):
    def __init__(self, prompts: List[str]):
        self.prompts = prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]


CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def _clip_preprocess_tensor(img_bchw: torch.Tensor, out_size: int = 224) -> torch.Tensor:
    """Differentiable resize + center-crop + CLIP normalization.
    img_bchw expected in [0,1]. Returns normalized tensor for CLIP.
    """
    b, c, h, w = img_bchw.shape
    # Resize so shorter side = out_size
    scale = out_size / min(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    img = F.interpolate(img_bchw, size=(nh, nw), mode="bilinear", align_corners=False)
    # Center crop
    top = max(0, (nh - out_size) // 2)
    left = max(0, (nw - out_size) // 2)
    img = img[:, :, top:top + out_size, left:left + out_size]
    # Normalize
    mean = CLIP_MEAN.to(img.device, img.dtype)
    std = CLIP_STD.to(img.device, img.dtype)
    img = (img - mean) / std
    return img


@dataclass
class PickScore:
    processor_name: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    model_name: str = "yuvalkirstain/PickScore_v1"

    def __post_init__(self):
        self.text_tokenizer = AutoTokenizer.from_pretrained(self.processor_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.eval()
        # Keep model params frozen
        for p in self.model.parameters():
            p.requires_grad_(False)

    def to(self, device):
        self.model.to(device)
        return self

    @torch.no_grad()
    def text_emb(self, texts: List[str], device: torch.device):
        tok = self.text_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(device)
        text_embs = self.model.get_text_features(**tok)
        text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)
        return text_embs

    def score(self, pixel_imgs: torch.Tensor, texts: List[str]) -> torch.Tensor:
        """Compute PickScore for a batch of images (B,3,H,W) in [0,1] and list of texts.
        Returns tensor of shape (B,) with differentiable path through the image.
        """
        device = pixel_imgs.device
        # Preprocess images differentiably
        px = _clip_preprocess_tensor(pixel_imgs)
        # Get image features (no grad in model params, but keep grad wrt inputs)
        image_embs = self.model.get_image_features(pixel_values=px)
        image_embs = image_embs / image_embs.norm(dim=-1, keepdim=True)
        # Text features (pre-computed or on-the-fly)
        tok = self.text_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            text_embs = self.model.get_text_features(**tok)
            text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)
        # Scale
        logit_scale = self.model.logit_scale.exp()
        # Cosine similarity
        scores = (logit_scale * (text_embs @ image_embs.T)).diagonal()
        return scores



def add_lora_to_unet(unet: UNet2DConditionModel, rank: int) -> AttnProcsLayers:
    lora_attn_procs = {}
    for name in unet.attn_processors.keys():
        # Determine dims for this processor
        if name.endswith("attn1.processor"):
            cross_attention_dim = None
        else:
            cross_attention_dim = unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name.split(".")[1])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name.split(".")[1])
            hidden_size = unet.config.block_out_channels[block_id]
        else:
            continue
        lora_attn_procs[name] = LoRAAttnProcessor2_0(
            hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=rank
        )
    unet.set_attn_processor(lora_attn_procs)
    # Return a convenience container with trainable params
    lora_layers = AttnProcsLayers(unet.attn_processors)
    return lora_layers


@dataclass
class SD15:
    model_name: str
    device: torch.device
    dtype: torch.dtype

    def __post_init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(self.model_name, subfolder="text_encoder")
        self.vae = AutoencoderKL.from_pretrained(self.model_name, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(self.model_name, subfolder="unet")
        self.scheduler = DDIMScheduler.from_pretrained(self.model_name, subfolder="scheduler")

        self.text_encoder.to(self.device, dtype=self.dtype)
        self.vae.to(self.device, dtype=self.dtype)
        self.unet.to(self.device, dtype=self.dtype)

        self.vae_scale_factor = 0.18215

    def encode_text(self, prompts: List[str]):
        tok = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            enc = self.text_encoder(**tok)
        return enc.last_hidden_state

    def get_uncond_emb(self, batch_size: int):
        return self.encode_text([""] * batch_size)

    @torch.no_grad()
    def vae_decode(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents / self.vae_scale_factor
        imgs = self.vae.decode(latents).sample
        imgs = (imgs.clamp(-1, 1) + 1) / 2  # to [0,1]
        return imgs

    def vae_encode(self, pixel_imgs: torch.Tensor) -> torch.Tensor:
        # pixel_imgs expected in [0,1]
        imgs = (pixel_imgs * 2 - 1).clamp(-1, 1)
        latents = self.vae.encode(imgs).latent_dist.mode()
        latents = latents * self.vae_scale_factor
        return latents


# -----------------------------
# DRaFT sampling & training core
# -----------------------------
@dataclass
class DraftConfig:
    ddim_steps: int = 50           
    guidance_scale: float = 7.5    
    method: str = "draft-k"       # {draft, draft-k, draft-lv}
    K: int = 1                    
    n_lv: int = 2                  
    gradient_checkpointing: bool = True


def ddim_timesteps_for(scheduler: DDIMScheduler, steps: int) -> torch.LongTensor:
    scheduler.set_timesteps(steps)
    return scheduler.timesteps


def sd_forward_unroll(
    sd: SD15,
    prompts: List[str],
    cfg: DraftConfig,
    generator: Optional[torch.Generator] = None,
    enable_grad: bool = True,
):
    device = sd.unet.device
    dtype = sd.unet.dtype
    bsz = len(prompts)

    # Text embeddings (cond & uncond)
    text_emb = sd.encode_text(prompts)
    uncond_emb = sd.get_uncond_emb(bsz)
    context = torch.cat([uncond_emb, text_emb], dim=0)

    timesteps = ddim_timesteps_for(sd.scheduler, cfg.ddim_steps)

    # Init latents
    shape = (bsz, sd.unet.in_channels, sd.vae.config.sample_size, sd.vae.config.sample_size)
    shape = (shape[0], shape[1], shape[2] // 8, shape[3] // 8)  # latent resolution
    latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)

    # Truncation boundary for DRaFT-K
    if cfg.method == "draft":
        k_boundary = 0
    else:
        k = max(1, min(cfg.K, cfg.ddim_steps))
        k_boundary = len(timesteps) - k

    # Enable gradient checkpointing at the module level (saves memory for full DRaFT)
    if cfg.gradient_checkpointing:
        sd.unet.enable_gradient_checkpointing()
    else:
        sd.unet.disable_gradient_checkpointing()

    # Unroll sampling
    for i, t in enumerate(timesteps):
        do_grad = enable_grad and (i >= k_boundary)
        ctx = torch.enable_grad() if do_grad else torch.no_grad()
        with ctx:
            # Classifier-free guidance: duplicate latents
            latent_in = torch.cat([latents] * 2, dim=0)
            # Scale model input (diffusers schedulers often scale by sigma)
            latent_in = sd.scheduler.scale_model_input(latent_in, t)
            noise_pred = sd.unet(
                latent_in, t, encoder_hidden_states=context
            ).sample
            noise_uncond, noise_text = noise_pred.chunk(2)
            noise_pred = noise_uncond + cfg.guidance_scale * (noise_text - noise_uncond)
            latents = sd.scheduler.step(noise_pred, t, latents).prev_sample
    # latents is z_0
    pixels = sd.vae_decode(latents)
    return pixels, latents, timesteps


def draft_lv_last_step(
    sd: SD15,
    x0_pixels: torch.Tensor,
    prompts: List[str],
    timesteps: torch.LongTensor,
    n_lv: int,
    guidance_scale: float,
):
    """Low-variance estimator for K=1: re-noise x0 to t_1 (the last used timestep),
    do one denoising step multiple times with different noises and sum rewards.
    Returns list of pixel tensors (one per LV inner loop).
    """
    assert n_lv >= 1
    device = sd.unet.device
    bsz = x0_pixels.shape[0]
    # Encode the clean images to latents
    z0 = sd.vae_encode(x0_pixels)
    t_last = timesteps[-1]  # the smallest timestep (last iteration)

    outs = []
    for _ in range(n_lv):
        noise = torch.randn_like(z0)
        zt = sd.scheduler.add_noise(z0, noise, t_last)
        latent_in = torch.cat([zt] * 2, dim=0)
        latent_in = sd.scheduler.scale_model_input(latent_in, t_last)
        text_emb = sd.encode_text(prompts)
        uncond_emb = sd.get_uncond_emb(bsz)
        context = torch.cat([uncond_emb, text_emb], dim=0)
        noise_pred = sd.unet(latent_in, t_last, encoder_hidden_states=context).sample
        noise_uncond, noise_text = noise_pred.chunk(2)
        noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)
        z_prev = sd.scheduler.step(noise_pred, t_last, zt).prev_sample
        x_pixels = sd.vae_decode(z_prev)
        outs.append(x_pixels)
    return outs


# -----------------------------
# Sampling to disk (train/val/final)
# -----------------------------

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def save_samples(
    sd: SD15,
    out_root: Path,
    split: str,
    step: int,
    prompts: List[str],
    num_images_per_prompt: int,
    cfg: DraftConfig,
    seed: int,
    width: int,
    height: int,
):
    if not prompts:
        return
    folder = out_root / split / f"step_{step:07d}"
    _ensure_dir(folder)
    csv_path = folder / "index.csv"
    write_header = not csv_path.exists()

    device = sd.unet.device
    gen = torch.Generator(device=device).manual_seed(seed + step)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["step", "split", "prompt_idx", "sample_idx", "filename", "prompt"])
        for p_idx, prompt in enumerate(prompts):
            for s_idx in range(num_images_per_prompt):
                pixels, _, _ = sd_forward_unroll(
                    sd, [prompt], cfg=cfg, generator=gen, enable_grad=False
                )
                img = (pixels[0].clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
                ts = now_utc()
                fn = f"{ts}_p{p_idx:05d}_{s_idx:02d}.png"
                fp = folder / fn
                Image.fromarray(img).save(fp)
                writer.writerow([step, split, p_idx, s_idx, str(fp), prompt])


import argparse

def read_prompts_file(path: Optional[str]) -> List[str]:
    if not path:
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def load_train_val_prompts(args) -> Tuple[List[str], List[str]]:
    if args.train_prompts_file or args.val_prompts_file:
        train_prompts = read_prompts_file(args.train_prompts_file)
        val_prompts = read_prompts_file(args.val_prompts_file)
    elif args.prompts_file:
        all_prompts = read_prompts_file(args.prompts_file)
        n_val = min(args.num_val_prompts, len(all_prompts))
        val_prompts = all_prompts[:n_val]
        train_prompts = all_prompts[n_val:]
    else:
        train_prompts, val_prompts = [], []
    return train_prompts, val_prompts


def parse_args():
    p = argparse.ArgumentParser(
        description="DRaFT/DRaFT-K/DRaFT-LV finetuning for SD1.5 with PickScore"
    )

    p.add_argument("--pretrained_model_name_or_path", type=str, default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--output_dir", type=str, default="outputs")

    p.add_argument("--ddim_steps", type=int, default=50, help="Sampling steps (50 per paper)")
    p.add_argument("--guidance_scale", type=float, default=7.5, help="CFG guidance (7.5 per paper)")


    p.add_argument("--method", type=str, choices=["draft", "draft-k", "draft-lv"], default="draft-k")
    p.add_argument("--K", type=int, default=1, help="Backprop through last-K steps (draft-k)")
    p.add_argument("--n_lv", type=int, default=2, help="DRaFT-LV inner loops (paper uses 2)")


    p.add_argument("--train_steps", type=int, default=2000, help="2k small-scale; 10k large-scale")
    p.add_argument("--batch_size", type=int, default=4, help="4 small-scale; 16 large-scale")
    p.add_argument("--lr", type=float, default=4e-4, help="4e-4 small-scale; 2e-4 large-scale")
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--adam_beta1", type=float, default=0.9)
    p.add_argument("--adam_beta2", type=float, default=0.999)

    p.add_argument("--lora_rank", type=int, default=8, help="LoRA inner dim (8 small-scale; 32 large-scale)")

    p.add_argument("--train_prompts_file", type=str, default=None)
    p.add_argument("--val_prompts_file", type=str, default=None)
    p.add_argument("--prompts_file", type=str, default=None)
    p.add_argument("--num_val_prompts", type=int, default=100)

    p.add_argument("--sample_every", type=int, default=200)
    p.add_argument("--samples_per_prompt", type=int, default=1)
    p.add_argument("--save_samples_dir", type=str, default="samples")

   
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--no_gradient_checkpointing", action="store_true")
    p.add_argument("--mixed_precision", type=str, choices=["no", "fp16", "bf16"], default="no")
    return p.parse_args()


# -----------------------------
# Main training
# -----------------------------

def main():
    args = parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Mixed precision
    dtype = torch.float32
    if args.mixed_precision == "bf16" and torch.cuda.is_available():
        dtype = torch.bfloat16
    elif args.mixed_precision == "fp16" and torch.cuda.is_available():
        dtype = torch.float16

    # Build SD1.5
    sd = SD15(args.pretrained_model_name_or_path, device=device, dtype=dtype)

    # LoRA
    lora = add_lora_to_unet(sd.unet, rank=args.lora_rank)
    # Ensure LoRA params live on same device/dtype as UNet
    lora.to(device=device, dtype=sd.unet.dtype)

    # PickScore
    pick = PickScore().to(device)

    # Optimizer (AdamW)
    opt = torch.optim.AdamW(
        lora.parameters(),
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
    )

    # Draft config
    draft_cfg = DraftConfig(
        ddim_steps=args.ddim_steps,
        guidance_scale=args.guidance_scale,
        method=args.method,
        K=args.K,
        n_lv=args.n_lv,
        gradient_checkpointing=(args.gradient_checkpointing and not args.no_gradient_checkpointing),
    )

    # Prompts
    train_prompts, val_prompts = load_train_val_prompts(args)
    train_ds = PromptDataset(train_prompts) if train_prompts else PromptDataset(["a photo of a dog"])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples_dir = Path(args.save_samples_dir)
    samples_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    global_step = 0
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision == "fp16"))

    while global_step < args.train_steps:
        for batch_prompts in train_loader:
            if global_step >= args.train_steps:
                break

            # Forward unroll (with gradients in the last-K steps according to method)
            gen = torch.Generator(device=device).manual_seed(args.seed + global_step)
            autocast = torch.cuda.amp.autocast if args.mixed_precision in {"fp16", "bf16"} else torch.cpu.amp.autocast

            with autocast(enabled=(args.mixed_precision != "no"), dtype=dtype):
                pixels, z0, timesteps = sd_forward_unroll(
                    sd, list(batch_prompts), cfg=draft_cfg, generator=gen, enable_grad=True
                )

                if args.method == "draft-lv":
                    if draft_cfg.K != 1:
                        raise ValueError("DRaFT-LV requires K=1 (last-step).")
                    # Average rewards over n_lv re-noisings of x0
                    inner_imgs = draft_lv_last_step(
                        sd, pixels.detach(), list(batch_prompts), timesteps, draft_cfg.n_lv, draft_cfg.guidance_scale
                    )
                    rewards = []
                    for x in inner_imgs:
                        r = pick.score(x, list(batch_prompts))
                        rewards.append(r)
                    reward = torch.stack(rewards, dim=0).mean(dim=0)
                else:
                    reward = pick.score(pixels, list(batch_prompts))

                loss = -reward.mean()

            # Backward + step
            opt.zero_grad(set_to_none=True)
            if args.mixed_precision == "fp16":
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()

            global_step += 1

            # Sampling hook
            if (global_step % args.sample_every == 0) or (global_step == 1):
                try:
                    save_samples(
                        sd,
                        out_root=samples_dir,
                        split="train",
                        step=global_step,
                        prompts=train_prompts[: min(16, len(train_prompts))],
                        num_images_per_prompt=args.samples_per_prompt,
                        cfg=draft_cfg,
                        seed=args.seed,
                        width=512,
                        height=512,
                    )
                    save_samples(
                        sd,
                        out_root=samples_dir,
                        split="val",
                        step=global_step,
                        prompts=val_prompts[: min(16, len(val_prompts))],
                        num_images_per_prompt=args.samples_per_prompt,
                        cfg=draft_cfg,
                        seed=args.seed + 1,
                        width=512,
                        height=512,
                    )
                except Exception as e:
                    print(f"[WARN] Sampling failed at step {global_step}: {e}")

            if global_step % 50 == 0:
                print(f"step={global_step} loss={loss.item():.4f}")

    # Final samples
    save_samples(
        sd,
        out_root=samples_dir,
        split="final",
        step=global_step,
        prompts=val_prompts[: min(16, len(val_prompts))] or train_prompts[: min(16, len(train_prompts))],
        num_images_per_prompt=args.samples_per_prompt,
        cfg=draft_cfg,
        seed=args.seed + 999,
        width=512,
        height=512,
    )

    # Save LoRA weights
    lora_path = out_dir / f"lora_unet_rank{args.lora_rank}_steps{args.train_steps}.pt"
    torch.save(lora.state_dict(), lora_path)
    print(f"Saved LoRA weights to {lora_path}")


if __name__ == "__main__":
    main()
