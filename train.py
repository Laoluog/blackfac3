"""Train a face recognition backbone (AdaFace / ArcFace / CosFace head) on a
balanced corpus (BUPT-Balancedface) for skin-tone-fair face recognition.

This is a transparent trainer that reuses the AdaFace repository's backbone
(``net.build_model``) and margin head (``head.AdaFace`` / ``ArcFace`` /
``CosFace``) verbatim, but runs a plain PyTorch loop so there is no opaque
validation plumbing to crash a run mid-epoch. Checkpoints are saved in a
format that ``evaluate_rfw.py`` loads directly.

Data sources (choose with --data-format):
  * mxrec       : InsightFace train.rec/train.idx (needs mxnet; Python <=3.10)
  * imagefolder : a directory of identity subfolders of images (no mxnet)

Fairness experiment hooks (--reweight):
  * none : standard uniform sampling
  * race : inverse-frequency over BUPT race label (needs --lst)
  * ita  : inverse-frequency over percentile ITA bins (imagefolder mode;
           continuous skin tone rather than 4 coarse race buckets)
"""
from __future__ import annotations

import argparse
import inspect
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import ImageFile
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

import bupt_labels

# The BUPT archive can be truncated at the extraction boundary, leaving a few
# partially-written JPEGs. Let PIL load what it can instead of crashing a run.
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger."""
    logging.basicConfig(
        level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
class MXFaceDataset(Dataset):
    """Reads an InsightFace train.rec/train.idx pair (mxnet RecordIO).

    Copied from the AdaFace repo's data.py so behavior matches the official
    pipeline. Images are decoded RGB and normalized to [-1, 1].
    """

    def __init__(self, root_dir: str, hflip: bool = True) -> None:
        import mxnet as mx  # local import: only needed for mxrec mode
        from torchvision import transforms

        tlist = [transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)]
        if hflip:
            tlist.insert(0, transforms.RandomHorizontalFlip())
        self.transform = transforms.Compose(tlist)
        path_imgrec = str(Path(root_dir) / "train.rec")
        path_imgidx = str(Path(root_dir) / "train.idx")
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, "r")
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        self.num_classes: int | None = None
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
            # InsightFace recs append identity-range records after the images:
            # header0 = (num_images+1, num_images+num_classes+1).
            n_cls = self.header0[1] - self.header0[0]
            self.num_classes = int(n_cls) if n_cls > 0 else None
        else:
            self.imgidx = np.array(list(self.imgrec.keys))
        self._mx = mx

    def __getitem__(self, index: int):
        import numbers

        from PIL import Image

        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = self._mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        sample = self._mx.image.imdecode(img).asnumpy()
        sample = Image.fromarray(sample)
        return self.transform(sample), label

    def __len__(self) -> int:
        return len(self.imgidx)


def build_imagefolder(root: str, hflip: bool = True):
    """Build a torchvision ImageFolder with the standard [-1,1] normalization."""
    from torchvision import datasets, transforms

    # Force a uniform 112x112 so batches collate even when a few images decode
    # at an odd size (e.g. partially-written/truncated JPEGs).
    tlist = [
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ]
    if hflip:
        tlist.insert(0, transforms.RandomHorizontalFlip())
    tf = transforms.Compose(tlist)
    # Some identity dirs can be empty (a truncated/quota-limited extraction left
    # only zero-byte files that were then deleted). torchvision raises on empty
    # classes unless allow_empty=True (added in torchvision 0.18); tolerate both.
    try:
        return datasets.ImageFolder(root, transform=tf, allow_empty=True)
    except TypeError:
        return datasets.ImageFolder(root, transform=tf)


# --------------------------------------------------------------------------- #
# Model / head
# --------------------------------------------------------------------------- #
def add_repo_to_path(adaface_repo: str) -> None:
    """Insert the AdaFace repo on sys.path so ``net``/``head`` import."""
    repo = str(Path(adaface_repo).expanduser().resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)


def build_backbone(arch: str, adaface_repo: str) -> torch.nn.Module:
    """Construct the IResNet backbone via the AdaFace repo's net.build_model."""
    add_repo_to_path(adaface_repo)
    import net  # type: ignore

    return net.build_model(arch)


def build_head(
    head_name: str,
    embedding_size: int,
    num_classes: int,
    m: float,
    h: float,
    s: float,
    t_alpha: float,
    adaface_repo: str,
) -> tuple[torch.nn.Module, bool]:
    """Construct the margin head from the AdaFace repo.

    Returns (head_module_instance, uses_norm) where ``uses_norm`` indicates
    whether the head's forward signature accepts the per-sample feature norm
    (AdaFace does; ArcFace/CosFace typically ignore it).
    """
    add_repo_to_path(adaface_repo)
    import head as head_mod  # type: ignore

    name = head_name.lower()
    if name == "adaface":
        head = head_mod.AdaFace(embedding_size, num_classes, m, h, s, t_alpha)
    elif name == "arcface":
        head = head_mod.ArcFace(embedding_size, num_classes, m=m, s=s)
    elif name == "cosface":
        head = head_mod.CosFace(embedding_size, num_classes, m=m, s=s)
    else:
        raise ValueError(f"Unknown head: {head_name}")
    params = inspect.signature(head.forward).parameters
    uses_norm = len(params) >= 3  # (embeddings, norms, label)
    return head, uses_norm


# --------------------------------------------------------------------------- #
# Sampler
# --------------------------------------------------------------------------- #
def build_sampler(
    reweight: str,
    dataset: Dataset,
    lst_path: str | None,
    ita_json: str | None,
    n_bins: int,
) -> WeightedRandomSampler | None:
    """Return a WeightedRandomSampler implementing the chosen fairness reweighting.

    'none' -> None (uniform). 'race' -> inverse race frequency (needs --lst,
    order-matched to the rec). 'ita' -> inverse ITA-bin frequency over the
    ImageFolder's sample paths (uses cached ita_json when available).
    """
    if reweight == "none":
        return None

    if reweight == "race":
        if not lst_path:
            raise ValueError("--reweight race requires --lst")
        entries = bupt_labels.parse_lst(lst_path)
        if len(entries) != len(dataset):
            logger.warning(
                "lst length (%d) != dataset length (%d); race weights may be misaligned",
                len(entries), len(dataset),
            )
        w = bupt_labels.race_sample_weights(entries)

    elif reweight == "ita":
        if not hasattr(dataset, "samples"):
            raise ValueError("--reweight ita requires imagefolder mode (paths needed)")
        paths = [p for p, _ in dataset.samples]  # type: ignore[attr-defined]
        ita = None
        if ita_json and Path(ita_json).exists():
            with open(ita_json) as f:
                cache = json.load(f)
            ita = np.array([cache.get(p, np.nan) for p in paths], dtype=np.float64)
            if np.isnan(ita).all():
                logger.warning("ita_json had no matching paths; recomputing")
                ita = None
        if ita is None:
            logger.info("Computing ITA over %d training images...", len(paths))
            ita = bupt_labels.compute_ita_for_paths(paths)
            if ita_json:
                with open(ita_json, "w") as f:
                    json.dump({p: (None if np.isnan(v) else v) for p, v in zip(paths, ita)}, f)
        w = bupt_labels.ita_bin_weights(ita, n_bins)
    else:
        raise ValueError(f"Unknown reweight mode: {reweight}")

    return WeightedRandomSampler(
        weights=torch.as_tensor(w, dtype=torch.double),
        num_samples=len(w),
        replacement=True,
    )


# --------------------------------------------------------------------------- #
# Periodic RFW evaluation (optional)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def quick_rfw_eval(
    backbone: torch.nn.Module,
    device: torch.device,
    rfw_root: str,
    races: list[str],
) -> dict:
    """Run a fast RFW eval reusing evaluate_rfw's logic. Returns per-race metrics."""
    from evaluate_rfw import compute_metrics, compute_pair_scores, parse_pairs_file

    backbone.eval()
    out: dict = {}
    root = Path(rfw_root)
    for race in races:
        pf = root / "txts" / race / f"{race}_pairs.txt"
        if not pf.exists():
            logger.warning("RFW pairs missing for %s: %s", race, pf)
            continue
        pairs = parse_pairs_file(pf, race, root)
        scores, labels, _, _ = compute_pair_scores(backbone, device, pairs)
        if len(scores):
            out[race] = compute_metrics(scores, labels)
    backbone.train()
    return out


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def save_checkpoint(path: Path, backbone, head, optimizer, scheduler, epoch, args) -> None:
    """Persist a checkpoint compatible with evaluate_rfw.py (key: 'state_dict')."""
    torch.save(
        {
            "state_dict": backbone.state_dict(),
            "head_state_dict": head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "args": vars(args),
        },
        path,
    )


def train(args: argparse.Namespace) -> None:
    """Run the full training loop."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----- dataset -----
    if args.data_format == "mxrec":
        dataset: Dataset = MXFaceDataset(args.data_root, hflip=not args.no_flip)
    else:
        dataset = build_imagefolder(args.data_root, hflip=not args.no_flip)

    # ----- num classes -----
    if args.num_classes:
        num_classes = args.num_classes
    elif args.lst:
        num_classes = bupt_labels.num_classes_from_entries(bupt_labels.parse_lst(args.lst))
    elif getattr(dataset, "num_classes", None):
        num_classes = dataset.num_classes  # type: ignore[attr-defined]  # from rec header
    elif hasattr(dataset, "classes"):
        num_classes = len(dataset.classes)  # type: ignore[attr-defined]
    else:
        raise ValueError("Cannot infer --num-classes; pass it explicitly or provide --lst")
    logger.info("num_classes=%d, dataset_size=%d", num_classes, len(dataset))

    # ----- sampler / loader -----
    sampler = build_sampler(args.reweight, dataset, args.lst, args.ita_json, args.n_bins)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # ----- model + head -----
    backbone = build_backbone(args.arch, args.adaface_repo).to(device)
    head, uses_norm = build_head(
        args.head, args.embedding_size, num_classes,
        args.m, args.h, args.s, args.t_alpha, args.adaface_repo,
    )
    head = head.to(device)
    logger.info("head=%s uses_norm=%s", args.head, uses_norm)

    optimizer = torch.optim.SGD(
        list(backbone.parameters()) + list(head.parameters()),
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
    )
    milestones = [int(x) for x in args.lr_milestones.split(",")]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=args.lr_gamma
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    start_epoch = 0
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location="cpu")
        backbone.load_state_dict(ckpt["state_dict"])
        head.load_state_dict(ckpt["head_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        logger.info("Resumed from %s at epoch %d", args.resume, start_epoch)

    backbone.train()
    head.train()
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        running = 0.0
        correct = total = 0
        pbar = tqdm(loader, desc=f"epoch {epoch}")
        for step, (images, labels) in enumerate(pbar):
            if args.max_steps and step >= args.max_steps:
                logger.info("Hit --max-steps=%d (smoke test); ending epoch early", args.max_steps)
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp):
                emb, norm = backbone(images)
                logits = head(emb, norm, labels) if uses_norm else head(emb, labels)
                loss = F.cross_entropy(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running += loss.item()
            preds = logits.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.numel()
            pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{correct / max(total,1):.3f}")
        scheduler.step()
        logger.info(
            "epoch %d done in %.1fs avg_loss=%.4f train_acc=%.4f lr=%.5f",
            epoch, time.time() - t0, running / max(len(loader), 1),
            correct / max(total, 1), scheduler.get_last_lr()[0],
        )

        ckpt_path = out_dir / f"{args.prefix}_epoch{epoch}.ckpt"
        save_checkpoint(ckpt_path, backbone, head, optimizer, scheduler, epoch, args)
        save_checkpoint(out_dir / f"{args.prefix}_last.ckpt", backbone, head, optimizer, scheduler, epoch, args)

        if args.rfw_root and args.eval_every and (epoch + 1) % args.eval_every == 0:
            metrics = quick_rfw_eval(backbone, device, args.rfw_root, args.eval_races)
            for race, m in metrics.items():
                logger.info("[val] epoch %d %s TAR@1%%=%.4f", epoch, race, m["TAR@FAR=0.01"])

    logger.info("Training complete. Final checkpoint: %s", out_dir / f"{args.prefix}_last.ckpt")


def parse_args() -> argparse.Namespace:
    """Build CLI parser."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # data
    p.add_argument("--data-root", required=True, help="mxrec dir (train.rec/idx) or imagefolder root")
    p.add_argument("--data-format", choices=["mxrec", "imagefolder"], default="mxrec")
    p.add_argument("--lst", default=None, help="BUPT train_balancedface.lst (for race reweight / num-classes)")
    p.add_argument("--num-classes", type=int, default=None)
    p.add_argument("--no-flip", action="store_true", help="disable horizontal flip aug")
    # model
    p.add_argument("--adaface-repo", required=True, help="Path to AdaFace repo (net.py, head.py)")
    p.add_argument("--arch", default="ir_50", help="ir_50, ir_101, ...")
    p.add_argument("--head", default="adaface", choices=["adaface", "arcface", "cosface"])
    p.add_argument("--embedding-size", type=int, default=512)
    p.add_argument("--m", type=float, default=0.4)
    p.add_argument("--h", type=float, default=0.333)
    p.add_argument("--s", type=float, default=64.0)
    p.add_argument("--t-alpha", type=float, default=1.0)
    # fairness experiment
    p.add_argument("--reweight", choices=["none", "race", "ita"], default="none")
    p.add_argument("--ita-json", default=None, help="cache of per-training-image ITA (ita reweight)")
    p.add_argument("--n-bins", type=int, default=5)
    # optim
    p.add_argument("--epochs", type=int, default=26)
    p.add_argument("--max-steps", type=int, default=0, help="stop each epoch after N batches (0=off; for smoke tests)")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--lr-milestones", default="12,20,24")
    p.add_argument("--lr-gamma", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--amp", action="store_true", help="mixed precision (recommended on H100)")
    # io / eval
    p.add_argument("--output-dir", required=True)
    p.add_argument("--prefix", default="bupt_adaface")
    p.add_argument("--resume", default=None)
    p.add_argument("--rfw-root", default=None, help="enable periodic RFW eval")
    p.add_argument("--eval-every", type=int, default=0, help="epochs between RFW evals (0=off)")
    p.add_argument("--eval-races", nargs="+", default=["African", "Caucasian"])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    """CLI entry point."""
    setup_logging()
    train(parse_args())


if __name__ == "__main__":
    main()
