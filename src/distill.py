"""Fairness-preserving knowledge distillation for edge deployment.

Distills a trained teacher backbone (e.g. IR-101) into a smaller student
(e.g. IR-50) for edge/on-device use. Compression tends to *amplify*
demographic bias, so this trainer supports weighting the distillation loss
by ITA bin (continuous skin tone) to preserve performance on darker-skinned
faces under compression.

Loss = (1 - alpha) * CE(student_head)              [optional, if --num-classes]
       + alpha * (1 - cos(student_emb, teacher_emb))  [feature distillation]
weighted per-sample by ITA-bin weight when --reweight ita is set.

EXPERIMENTAL: syntax/CLI verified locally; not yet run on real data/GPU.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import bupt_labels
from train import (
    MXFaceDataset,
    add_repo_to_path,
    build_backbone,
    build_imagefolder,
    build_sampler,
    save_checkpoint,
    setup_logging,
)

logger = logging.getLogger(__name__)


def load_teacher(arch: str, ckpt: str, adaface_repo: str, device: torch.device) -> torch.nn.Module:
    """Build teacher backbone, load weights, freeze, eval-mode."""
    model = build_backbone(arch, adaface_repo)
    state = torch.load(ckpt, map_location="cpu")
    sd = state.get("state_dict", state) if isinstance(state, dict) else state
    sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        logger.warning("Teacher missing keys: %s", missing)
    model = model.eval().to(device)
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def distill(args: argparse.Namespace) -> None:
    """Run the distillation loop."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.data_format == "mxrec":
        dataset = MXFaceDataset(args.data_root, hflip=not args.no_flip)
    else:
        dataset = build_imagefolder(args.data_root, hflip=not args.no_flip)

    sampler = build_sampler(args.reweight, dataset, args.lst, args.ita_json, args.n_bins)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler,
        shuffle=(sampler is None), num_workers=args.num_workers,
        pin_memory=True, drop_last=True,
    )

    teacher = load_teacher(args.teacher_arch, args.teacher_ckpt, args.adaface_repo, device)
    student = build_backbone(args.student_arch, args.adaface_repo).to(device)
    student.train()

    optimizer = torch.optim.SGD(
        student.parameters(), lr=args.lr, momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    milestones = [int(x) for x in args.lr_milestones.split(",")]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=args.lr_gamma)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    for epoch in range(args.epochs):
        t0 = time.time()
        running = 0.0
        for images, _ in tqdm(loader, desc=f"distill epoch {epoch}"):
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp):
                with torch.no_grad():
                    t_emb, _ = teacher(images)
                s_emb, _ = student(images)
                # both embeddings are L2-normalized by the backbone
                cos = F.cosine_similarity(s_emb, t_emb, dim=1)
                loss = (1.0 - cos).mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
        scheduler.step()
        logger.info(
            "distill epoch %d done in %.1fs avg_loss=%.4f lr=%.5f",
            epoch, time.time() - t0, running / max(len(loader), 1),
            scheduler.get_last_lr()[0],
        )
        save_checkpoint(
            out_dir / f"{args.prefix}_epoch{epoch}.ckpt",
            student, torch.nn.Identity(), optimizer, scheduler, epoch, args,
        )
        save_checkpoint(
            out_dir / f"{args.prefix}_last.ckpt",
            student, torch.nn.Identity(), optimizer, scheduler, epoch, args,
        )
    logger.info("Distillation complete: %s", out_dir / f"{args.prefix}_last.ckpt")


def parse_args() -> argparse.Namespace:
    """Build CLI parser."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-root", required=True)
    p.add_argument("--data-format", choices=["mxrec", "imagefolder"], default="mxrec")
    p.add_argument("--lst", default=None)
    p.add_argument("--no-flip", action="store_true")
    p.add_argument("--adaface-repo", required=True)
    p.add_argument("--teacher-arch", default="ir_101")
    p.add_argument("--teacher-ckpt", required=True)
    p.add_argument("--student-arch", default="ir_50")
    p.add_argument("--reweight", choices=["none", "race", "ita"], default="none")
    p.add_argument("--ita-json", default=None)
    p.add_argument("--n-bins", type=int, default=5)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--lr-milestones", default="10,16,18")
    p.add_argument("--lr-gamma", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--prefix", default="distilled_student")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    """CLI entry point."""
    setup_logging()
    distill(parse_args())


if __name__ == "__main__":
    main()
