import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from lightly.loss import NTXentLoss
from lightly.models.modules.heads import MoCoProjectionHead

sys.path.insert(0, os.getcwd())

from models.C3DAVG.C3D_altered import C3D_altered
from ssl_dataloader import SSLVideoClipDataset


class C3DMoCo(nn.Module):
    def __init__(self, feat_dim: int = 8192, proj_hidden: int = 2048, proj_out: int = 128):
        super().__init__()
        self.backbone = C3D_altered()
        self.projection_head = MoCoProjectionHead(feat_dim, proj_hidden, proj_out)

    def forward(self, x):
        feats = self.backbone(x)
        z = self.projection_head(feats)
        z = nn.functional.normalize(z, dim=1)
        return feats, z


def load_c3d_base(model: C3DMoCo, ckpt_path: str):
    if not ckpt_path or not os.path.isfile(ckpt_path):
        print(f'[WARN] Base checkpoint not found: {ckpt_path}')
        return

    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']

    backbone_state = model.backbone.state_dict()
    matched = {
        k: v for k, v in state.items()
        if k in backbone_state and backbone_state[k].shape == v.shape
    }
    backbone_state.update(matched)
    model.backbone.load_state_dict(backbone_state)
    print(f'[INFO] Loaded {len(matched)} matching backbone tensors from {ckpt_path}')


def save_checkpoint(path, model, optimizer, epoch, args, best_loss):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'backbone_state_dict': model.backbone.state_dict(),
            'projection_head_state_dict': model.projection_head.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss,
            'args': vars(args),
        },
        path,
    )


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    count = 0

    for view1, view2, _ in loader:
        view1 = view1.to(device, non_blocking=True)
        view2 = view2.to(device, non_blocking=True)

        _, z1 = model(view1)
        _, z2 = model(view2)
        loss = criterion(z1, z2)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        bs = view1.size(0)
        running_loss += loss.item() * bs
        count += bs

    return running_loss / max(count, 1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--frames-root', type=str, required=True)
    p.add_argument('--c3d-base', type=str, default='')
    p.add_argument('--output-dir', type=str, required=True)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--clip-len', type=int, default=16)
    p.add_argument('--stride', type=int, default=1)
    p.add_argument('--proj-hidden', type=int, default=2048)
    p.add_argument('--proj-out', type=int, default=128)
    p.add_argument('--save-every', type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[INFO] Device: {device}')

    dataset = SSLVideoClipDataset(
        frames_root=args.frames_root,
        clip_len=args.clip_len,
        stride=args.stride,
        min_frames=args.clip_len,
        size=112,
    )
    print(f'[INFO] Videos discovered: {len(dataset)}')

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    model = C3DMoCo(
        proj_hidden=args.proj_hidden,
        proj_out=args.proj_out
    ).to(device)

    load_c3d_base(model, args.c3d_base)

    criterion = NTXentLoss(temperature=0.2)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    os.makedirs(args.output_dir, exist_ok=True)
    best_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        avg_loss = train_one_epoch(model, loader, criterion, optimizer, device)
        print(f'[EPOCH {epoch:03d}] loss={avg_loss:.6f}')

        latest_path = os.path.join(args.output_dir, 'c3d_ssl_latest.pth')
        save_checkpoint(latest_path, model, optimizer, epoch, args, best_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(args.output_dir, 'c3d_ssl_best.pth')
            save_checkpoint(best_path, model, optimizer, epoch, args, best_loss)

        if epoch % args.save_every == 0:
            ep_path = os.path.join(args.output_dir, f'c3d_ssl_epoch_{epoch:03d}.pth')
            save_checkpoint(ep_path, model, optimizer, epoch, args, best_loss)

    backbone_only = os.path.join(args.output_dir, 'c3d_ssl_backbone_only.pth')
    torch.save(model.backbone.state_dict(), backbone_only)
    print(f'[DONE] Saved backbone-only weights to {backbone_only}')


if __name__ == '__main__':
    main()