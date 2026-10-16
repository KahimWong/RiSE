"""Train fixed-assignment RiSE experts with AdamW, AMP and optional torchrun DDP."""

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import random
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from rise.config import ModelConfig
from rise.checkpoint import load_model
from rise.data import ManifestDataset
from rise.models import RiSE
from rise.models.loss import MaskClassificationLoss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--output", default="outputs/train")
    parser.add_argument(
        "--resume",
        help="Full training checkpoint, including optimizer and scheduler state",
    )
    parser.add_argument(
        "--steps", type=int, default=10000, help="Optimizer updates; not microbatches"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Images per GPU per microbatch"
    )
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.accumulation, args.save_every) < 1:
        parser.error("steps, batch-size, accumulation, save-every must be positive")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world > 1:
        dist.init_process_group("nccl")
    rank = dist.get_rank() if world > 1 else 0
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    config = ModelConfig.read(args.config)
    # SVD initialization is expensive but only runs when starting fresh training.
    if args.resume:
        model, _ = load_model(args.resume, config, device)
        model.dct_fuse_prob = config.dct_fuse_prob
    else:
        model = RiSE(config, initialize=True).to(device)
    if world > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    raw_model = model.module if world > 1 else model
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.steps, eta_min=args.min_lr
    )
    scaler = torch.amp.GradScaler("cuda")
    criterion = MaskClassificationLoss().to(device)
    first_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint.get("train_args", {}).get("steps") != args.steps:
            raise ValueError(
                "Resume with the original --steps to preserve the cosine schedule"
            )
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        first_step = checkpoint["step"]
    # Epoch = one deterministic pass through the manifest; shuffle changes each epoch.
    dataset = ManifestDataset(
        args.manifest,
        args.data_root,
        "train",
        config.use_dct,
        num_experts=config.num_experts,
        training=True,
        min_quality=100,
    )
    sampler = (
        DistributedSampler(dataset, shuffle=True, seed=args.seed)
        if world > 1
        else RandomSampler(dataset)
    )
    generator = torch.Generator().manual_seed(args.seed + rank)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        generator=generator,
    )
    if len(loader) == 0:
        raise ValueError(
            "Training manifest must contain at least one complete local batch"
        )
    epoch = 0
    iterator = iter(loader)
    output = Path(args.output)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        (output / "model_config.json").write_text(
            json.dumps(config.to_dict(), indent=2)
        )
        (output / "train_args.json").write_text(json.dumps(vars(args), indent=2))
        print(
            f"Effective batch = {args.batch_size * args.accumulation * world}; optimizer updates = {args.steps}",
            flush=True,
        )
    # Resume restarts data shuffling; this is a training-state resume, not a bitwise replay.
    torch.manual_seed(args.seed + rank + first_step)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for step in range(first_step, args.steps):
        # JPEG curriculum: quality floor decreases by one every 200 updates to 90.
        dataset.min_quality.fill_(max(100 - step // 200, 90))
        total_loss = 0.0
        for micro in range(args.accumulation):
            try:
                batch = next(iterator)
            except StopIteration:
                epoch += 1
                if world > 1:
                    sampler.set_epoch(epoch)
                iterator = iter(loader)
                batch = next(iterator)
            image, mask = batch["image"].to(device), batch["mask"].to(device)
            ids = batch["expert_id"].to(device)
            inputs = {k: batch[k].to(device) for k in ("dct", "qt") if k in batch}
            targets = [
                {
                    "masks": m[None].float(),
                    "labels": torch.zeros(1, dtype=torch.long, device=device),
                }
                for m in mask
            ]
            sync = (
                model.no_sync()
                if world > 1 and micro < args.accumulation - 1
                else nullcontext()
            )
            with sync:
                with torch.autocast("cuda", dtype=torch.float16):
                    masks, classes, _ = model(image, ids, **inputs)
                # Deep supervision at the query blocks, retaining the original loss.
                losses = {}
                for block, (m, c) in enumerate(zip(masks, classes)):
                    losses.update(
                        {
                            f"{k}_{block}": v
                            for k, v in criterion(m.float(), targets, c.float()).items()
                        }
                    )
                loss = criterion.loss_total(losses)
                total_loss += float(loss.detach()) / args.accumulation
                scaler.scale(loss / args.accumulation).backward()
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= old_scale:
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        if rank == 0 and (step + 1) % 20 == 0:
            print(
                f"step={step+1} loss={total_loss:.5f} lr={scheduler.get_last_lr()[0]:.3g}",
                flush=True,
            )
        if rank == 0 and ((step + 1) % args.save_every == 0 or step + 1 == args.steps):
            torch.save(
                {
                    "model": raw_model.state_dict(),
                    "model_config": config.to_dict(),
                    "step": step + 1,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "train_args": vars(args),
                },
                output / f"step_{step+1:06d}.pth",
            )
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
