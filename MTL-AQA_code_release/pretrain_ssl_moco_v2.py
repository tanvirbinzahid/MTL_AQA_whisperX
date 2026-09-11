import argparse
import copy
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from lightly.models.modules import MoCoProjectionHead

sys.path.insert(0, os.getcwd())

from models.C3DAVG.C3D_altered import C3D_altered
from ssl_dataloader import SSLVideoClipDataset


class MocoC3D(nn.Module):
    def __init__(self, feat_dim=8192, proj_hidden=1024, proj_out=128):
        super().__init__()
        self.encoder_q = C3D_altered()
        self.encoder_k = C3D_altered()
        self.projection_head_q = MoCoProjectionHead(feat_dim, proj_hidden, proj_out)
        self.projection_head_k = MoCoProjectionHead(feat_dim, proj_hidden, proj_out)
        for p in self.encoder_k.parameters():
            p.requires_grad = False
        for p in self.projection_head_k.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def _momentum_update_key_encoder(self, m):
        for q, k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            k.data = k.data * m + q.data * (1.0 - m)
        for q, k in zip(self.projection_head_q.parameters(), self.projection_head_k.parameters()):
            k.data = k.data * m + q.data * (1.0 - m)

    @torch.no_grad()
    def _initialize_key_encoder(self):
        self.encoder_k.load_state_dict(copy.deepcopy(self.encoder_q.state_dict()))
        self.projection_head_k.load_state_dict(copy.deepcopy(self.projection_head_q.state_dict()))

    def forward(self, x_q, x_k):
        q = self.projection_head_q(self.encoder_q(x_q))
        q = F.normalize(q, dim=1)
        with torch.no_grad():
            k = self.projection_head_k(self.encoder_k(x_k))
            k = F.normalize(k, dim=1)
        return q, k


class MoCoQueue:
    def __init__(self, dim=128, K=1024):
        self.dim = dim
        self.K = K
        self.queue = F.normalize(torch.randn(K, dim), dim=1)
        self.ptr = 0

    @torch.no_grad()
    def to(self, device):
        self.queue = self.queue.to(device)
        return self

    @torch.no_grad()
    def dequeue_and_enqueue(self, keys):
        bs = keys.shape[0]
        if bs >= self.K:
            self.queue = F.normalize(keys[-self.K:].detach().clone(), dim=1)
            self.ptr = 0
            return
        end = self.ptr + bs
        if end <= self.K:
            self.queue[self.ptr:end] = keys.detach()
        else:
            first = self.K - self.ptr
            self.queue[self.ptr:] = keys[:first].detach()
            self.queue[:end - self.K] = keys[first:].detach()
        self.ptr = end % self.K


def moco_loss(q, k, queue, temperature=0.2):
    l_pos = torch.sum(q * k, dim=1, keepdim=True)
    l_neg = torch.mm(q, queue.t())
    logits = torch.cat([l_pos, l_neg], dim=1) / temperature
    labels = torch.zeros(q.size(0), dtype=torch.long, device=q.device)
    return F.cross_entropy(logits, labels)


def load_c3d_base(model, ckpt_path):
    if not ckpt_path or not os.path.isfile(ckpt_path):
        print(f'[WARN] Base checkpoint not found: {ckpt_path}')
        return
    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    backbone_state = model.encoder_q.state_dict()
    matched = {k: v for k, v in state.items() if k in backbone_state and backbone_state[k].shape == v.shape}
    backbone_state.update(matched)
    model.encoder_q.load_state_dict(backbone_state)
    model.encoder_k.load_state_dict(copy.deepcopy(model.encoder_q.state_dict()))
    print(f'[INFO] Loaded {len(matched)} tensors into query encoder from {ckpt_path}')


def save_checkpoint(path, model, optimizer, epoch, args, best_loss, queue):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_loss': best_loss,
        'queue': queue.queue,
        'queue_ptr': queue.ptr,
        'args': vars(args),
    }, path)


def train_one_epoch(model, loader, optimizer, queue, device, m, temperature):
    model.train()
    total = 0.0
    n = 0
    for x1, x2, _ in loader:
        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)
        q, k = model(x1, x2)
        loss = moco_loss(q, k, queue.queue, temperature=temperature)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            model._momentum_update_key_encoder(m)
            queue.dequeue_and_enqueue(k)
        bs = x1.size(0)
        total += loss.item() * bs
        n += bs
    return total / max(n, 1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--frames-root', type=str, required=True)
    p.add_argument('--c3d-base', type=str, default='')
    p.add_argument('--output-dir', type=str, required=True)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--clip-len', type=int, default=16)
    p.add_argument('--stride', type=int, default=1)
    p.add_argument('--proj-hidden', type=int, default=1024)
    p.add_argument('--proj-out', type=int, default=128)
    p.add_argument('--queue-size', type=int, default=1024)
    p.add_argument('--momentum', type=float, default=0.995)
    p.add_argument('--temperature', type=float, default=0.2)
    p.add_argument('--save-every', type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[INFO] Device: {device}')
    dataset = SSLVideoClipDataset(args.frames_root, clip_len=args.clip_len, stride=args.stride, min_frames=args.clip_len, size=112)
    print(f'[INFO] Videos discovered: {len(dataset)}')
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), drop_last=True, persistent_workers=args.num_workers > 0)
    model = MocoC3D(proj_hidden=args.proj_hidden, proj_out=args.proj_out).to(device)
    model._initialize_key_encoder()
    load_c3d_base(model, args.c3d_base)
    queue = MoCoQueue(dim=args.proj_out, K=args.queue_size).to(device)
    optimizer = torch.optim.Adam(model.encoder_q.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    os.makedirs(args.output_dir, exist_ok=True)
    best_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        avg_loss = train_one_epoch(model, loader, optimizer, queue, device, args.momentum, args.temperature)
        print(f'[EPOCH {epoch:03d}] loss={avg_loss:.6f} queue_ptr={queue.ptr}')
        save_checkpoint(os.path.join(args.output_dir, 'moco_c3d_latest.pth'), model, optimizer, epoch, args, best_loss, queue)
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(os.path.join(args.output_dir, 'moco_c3d_best.pth'), model, optimizer, epoch, args, best_loss, queue)
        if epoch % args.save_every == 0:
            save_checkpoint(os.path.join(args.output_dir, f'moco_c3d_epoch_{epoch:03d}.pth'), model, optimizer, epoch, args, best_loss, queue)
    torch.save(model.encoder_q.state_dict(), os.path.join(args.output_dir, 'moco_c3d_backbone_only.pth'))
    print(f'[DONE] Saved backbone-only weights to {os.path.join(args.output_dir, "moco_c3d_backbone_only.pth")}')


if __name__ == '__main__':
    main()
