#!/usr/bin/env python3
"""Non-interactive eval of the SSL-MoCo C3D MTL-AQA model on any diving dataset.

Loads SSL fine-tuned multi-part checkpoint from checkpoints/ (best_*):
  model_CNN, model_my_fc6, model_score_regressor, model_dive_classifier,
  model_caption. Forward: C3D sliding 16-frame clips -> fc6 mean ->
  score(reg) + cls(5) + caption (same as evaluate.py).

Usage:
  python3 eval_ssl_any.py <mtl|fine|amateur> [--ckpt DIR] [--epoch best|N] [--gpu N]
"""
import sys, os, argparse, pickle, json, glob, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from scipy import stats

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, "MTL-AQA_code_release"))

from models.C3DAVG.C3D_altered     import C3D_altered
from models.C3DAVG.my_fc6          import my_fc6
from models.C3DAVG.score_regressor import score_regressor
from models.C3DAVG.dive_classifier import dive_classifier
from models.C3DAVG.S2VTModel       import S2VTModel


def find_ckpt(ckpt_dir, name):
    p = Path(ckpt_dir) / name
    return p if p.exists() else None


def load_checkpoint_parts(ckpt_dir, kind):
    if kind == 'best':
        names = {
            'cnn':   ['best_model_CNN.pth'],
            'fc6':   ['best_model_my_fc6.pth'],
            'score': ['best_model_score_regressor.pth'],
            'cls':   ['best_model_dive_classifier.pth'],
            'cap':   ['best_model_caption.pth'],
        }
    else:
        names = {
            'cnn':   [f'model_CNN_{kind}.pth'],
            'fc6':   [f'model_my_fc6_{kind}.pth'],
            'score': [f'model_score_regressor_{kind}.pth'],
            'cls':   [f'model_dive_classifier_{kind}.pth'],
            'cap':   [f'model_caption_{kind}.pth'],
        }
    out = {}
    for k, cands in names.items():
        p = next((find_ckpt(ckpt_dir, c) for c in cands), None)
        if p is None:
            raise FileNotFoundError(f"missing {k} checkpoint")
        out[k] = p
    return out


def build_model(cap_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cnn       = C3D_altered().to(device)
    fc6       = my_fc6().to(device)
    score_reg = score_regressor().to(device)
    dive_cls  = dive_classifier().to(device)
    _cap_sd = torch.load(cap_path, map_location='cpu')
    vocab = _cap_sd['embedding.weight'].shape[0]
    dimvid = _cap_sd['rnn1.weight_ih_l0'].shape[1]
    caption_m = S2VTModel(vocab, max_len=128, dim_hidden=512, dim_word=512,
                          dim_vid=dimvid, n_layers=2, rnn_cell='gru',
                          rnn_dropout_p=0.5).to(device)
    return device, cnn, fc6, score_reg, dive_cls, caption_m


def load_all(paths):
    device, cnn, fc6, score_reg, dive_cls, caption_m = build_model(paths['cap'])
    cnn.load_state_dict(torch.load(paths['cnn'],   map_location=device))
    fc6.load_state_dict(torch.load(paths['fc6'],   map_location=device))
    score_reg.load_state_dict(torch.load(paths['score'], map_location=device))
    dive_cls.load_state_dict(torch.load(paths['cls'],   map_location=device))
    caption_m.load_state_dict(torch.load(paths['cap'],  map_location=device))
    cnn.eval(); fc6.eval(); score_reg.eval(); dive_cls.eval(); caption_m.eval()
    return device, cnn, fc6, score_reg, dive_cls, caption_m


# ---------------- dataset adapters ----------------
def get_adapter(name):
    if name == 'mtl':
        return MTLAdapter()
    if name == 'fine':
        return FineAdapter()
    if name == 'amateur':
        return AmateurAdapter()
    raise ValueError(name)


class MTLAdapter:
    def __init__(self, root=None, anno_path=None):
        self.dir = Path(root) if root else Path(BASE) / "MTL-AQA_dataset_release" / "frames"
        self.anno = pickle.load(open(anno_path if anno_path else os.path.join(
            BASE, "MTL-AQA_dataset_release", "Ready_2_Use", "MTL-AQA_split_0_data", "final_annotations_dict.pkl"), 'rb'))
        self.keys = pickle.load(open(os.path.join(
            BASE, "MTL-AQA_dataset_release", "Ready_2_Use", "MTL-AQA_split_0_data", "test_split_0.pkl"), 'rb'))
        self.sample_len = 96
        self.score_std = 17.0
        self.name = "MTL-AQA SSL"

    def frame_path(self, key, fidx):
        vid, dive = key
        return self.dir / f"{vid:02d}" / f"{fidx:06d}.jpg"

    def get_window(self, key):
        a = self.anno[key]
        end = a['end_frame']
        return list(range(end - self.sample_len, end))

    def true_attrs(self, key):
        a = self.anno[key]
        return {k: int(a[k]) for k in ['position', 'armstand', 'rotation_type', 'ss_no', 'tw_no']}

    def true_score(self, key):
        return float(self.anno[key]['final_score'])


class FineAdapter:
    def __init__(self):
        # FineDiving frames + annotations are NOT shipped in this repo.
        # Set FINEDIVING_ROOT / FINEDIVING_ANNO / FINEDIVING_TEST to your local copies.
        self.root = Path(os.environ.get("FINEDIVING_ROOT", "/home/tabz1225/projects/data/FINADiving_MTL_256s"))
        self.anno = pickle.load(open(os.environ.get(
            "FINEDIVING_ANNO", "/home/tabz1225/projects/HP-MCoRe/Annotations/fine-grained_annotation_aqa.pkl"), 'rb'))
        self.keys = pickle.load(open(os.environ.get(
            "FINEDIVING_TEST", "/home/tabz1225/projects/HP-MCoRe/Annotations/test_split.pkl"), 'rb'))
        self.sample_len = 96
        self.score_std = 17.0
        self.name = "FineDiving SSL"

    def frame_path(self, key, fidx):
        vname, round_id = key
        # find actual file by integer name (variable-length zero padding)
        d = self.root / str(vname) / str(round_id)
        g = glob.glob(str(d / '*.jpg'))
        names = {int(os.path.basename(f)[:-4]): f for f in g}
        f = names.get(int(fidx))
        return Path(f) if f else d / f'{int(fidx):06d}.jpg'

    def get_window(self, key):
        vname, round_id = key
        fps = sorted(glob.glob(str(self.root / str(vname) / str(round_id) / "*.jpg")))
        if len(fps) < 2:
            return None
        names = [int(os.path.basename(f)[:-4]) for f in fps]
        start = names[0]; end = names[-1]
        idx = np.linspace(start, end, self.sample_len).astype(int)
        nset = set(names)
        out = []
        for i in idx:
            if i in nset:
                out.append(i)
        return out if len(out) == self.sample_len else None

    def true_attrs(self, key):
        a = self.anno[key]
        dn = a[0].lower()
        pos_map = {'a': 0, 'b': 1, 'c': 2, 'd': 0}
        rot_map = {'1': 1, '2': 0, '3': 3, '4': 2}
        armstand = 1 if dn[0] == '6' else 0
        digits = dn[:-1]
        if digits[0] in ('5', '6'):
            rot_letter = digits[1]; rest = digits[2:]
        else:
            rot_letter = digits[0]; rest = digits[1:]
        rotation = rot_map.get(rot_letter, 0)
        ss_no = int(rest[0]) if rest else 0
        tw_no = int(rest[1]) if len(rest) >= 2 else 0
        return {'position': pos_map.get(dn[-1], 0), 'armstand': armstand,
                'rotation_type': rotation, 'ss_no': ss_no, 'tw_no': tw_no}

    def true_score(self, key):
        return float(self.anno[key][1])


class AmateurAdapter:
    def __init__(self):
        # AmateurDiving frames + manifest are NOT shipped in this repo.
        # Set AMATEUR_FRAMES / AMATEUR_VIEWS to your local copies.
        self.dir = Path(os.environ.get("AMATEUR_FRAMES", "/shared/rtis_lab/data/AQA/AmateurDiving/amateur_frames"))
        self.views = json.load(open(os.environ.get(
            "AMATEUR_VIEWS", "/shared/rtis_lab/data/AQA/AmateurDiving/amateur_views.json")))
        self.keys = list(range(len(self.views)))
        self.sample_len = 96
        self.score_std = 17.0
        self.name = "AmateurDive SSL"

    def frame_path(self, key, fidx):
        return self.dir / f"{key:04d}" / f"frame_{fidx:06d}.jpg"

    def get_window(self, key):
        return list(range(96))

    def true_attrs(self, key):
        v = self.views[key]
        dn = v['dive_number'].lower()
        pos_map = {'a': 0, 'b': 1, 'c': 2, 'd': 0}
        rot_map = {'1': 1, '2': 0, '3': 3, '4': 2}
        armstand = 1 if dn[0] == '6' else 0
        digits = dn[:-1]
        if digits[0] in ('5', '6'):
            rot_letter = digits[1]; rest = digits[2:]
        else:
            rot_letter = digits[0]; rest = digits[1:]
        rotation = rot_map.get(rot_letter, 0)
        ss_no = int(rest[0]) if rest else 0
        tw_no = int(rest[1]) if len(rest) >= 2 else 0
        rot_l = {'f': 1, 'b': 0, 'r': 3, 'i': 2}
        if v.get('rotation_type') and v['rotation_type'] in rot_l:
            rotation = rot_l[v['rotation_type']]
        return {'position': pos_map.get(dn[-1], 0), 'armstand': armstand,
                'rotation_type': rotation,
                'ss_no': int((v.get('n_sommersaults') or 0) * 2),
                'tw_no': int((v.get('n_twists') or 0) * 2)}

    def true_score(self, key):
        return float(self.views[key]['final_score'])


# ---------------- forward ----------------
def run_eval(adapter, device, cnn, fc6, score_reg, dive_cls, caption_m, idx2word):
    transform = transforms.Compose([
        transforms.Resize((128, 171)),
        transforms.CenterCrop(112),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    preds_score, trues_score = [], []
    preds_cls = [[] for _ in range(5)]
    trues_cls = [[] for _ in range(5)]
    n = len(adapter.keys)
    skipped = 0
    for i, key in enumerate(adapter.keys):
        win = adapter.get_window(key)
        if win is None:
            skipped += 1
            continue
        frames = []
        ok = True
        for f in win:
            p = adapter.frame_path(key, f)
            if not p.exists():
                ok = False
                break
            frames.append(transform(Image.open(str(p)).convert('RGB')))
        if not ok or len(frames) != adapter.sample_len:
            skipped += 1
            continue
        video = torch.stack(frames).permute(1, 0, 2, 3).unsqueeze(0).to(device)
        with torch.no_grad():
            clip_feats = torch.Tensor([]).to(device)
            for j in np.arange(0, adapter.sample_len - 17, 16):
                clip = video[:, :, int(j):int(j) + 16, :, :]
                clip_feats = torch.cat((clip_feats, cnn(clip)), 0)
            feat = fc6(clip_feats.mean(0).unsqueeze(0))
            pred_score = score_reg(feat).item() * adapter.score_std
            cls_out = dive_cls(feat)
            preds_score.append(pred_score)
            trues_score.append(adapter.true_score(key))
            ta = adapter.true_attrs(key)
            for j in range(5):
                preds_cls[j].append(cls_out[j].argmax(dim=1).item())
                trues_cls[j].append(ta[['position', 'armstand', 'rotation_type', 'ss_no', 'tw_no'][j]])
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n}", flush=True)
    ps = np.array(preds_score); ts = np.array(trues_score)
    rho, _ = stats.spearmanr(ps, ts)
    r, _ = stats.pearsonr(ps, ts)
    mse = np.mean((ps - ts) ** 2); mae = np.mean(np.abs(ps - ts))
    accs = [np.mean(np.array(preds_cls[j]) == np.array(trues_cls[j])) * 100 for j in range(5)]
    return {'rho': rho, 'pearson': r, 'mse': mse, 'mae': mae,
            'accs': accs, 'avg_acc': np.mean(accs), 'n': len(ps),
            'skipped': skipped, 'pred': ps, 'true': ts}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset', choices=['mtl', 'fine', 'amateur'])
    ap.add_argument('--ckpt', default=os.path.join(BASE, "best_nonssl/checkpoints"))
    ap.add_argument('--epoch', default='best')
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--root', default=None, help='frames root for MTL-AQA (default: <repo>/MTL-AQA_dataset_release/frames)')
    args = ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    paths = load_checkpoint_parts(args.ckpt, args.epoch)
    device, cnn, fc6, score_reg, dive_cls, caption_m = load_all(paths)

    vocab_candidates = [
        Path(f"{BASE}/MTL-AQA_dataset_release/Ready_2_Use/MTL-AQA_split_0_data/vocab.json"),
        Path(f"{BASE}/SSL/checkpoints/vocab.json"),
        Path(f"{BASE}/best_nonssl/checkpoints/vocab.json"),
    ]
    vocab_path = next((c for c in vocab_candidates if c.exists()), None)
    if vocab_path is None:
        raise FileNotFoundError("vocab.json not found")
    raw = json.load(open(vocab_path))
    raw_ixtow = raw.get('ixtoword', raw.get('ix_to_word', {}))
    if raw_ixtow:
        idx2word = {int(k): v for k, v in raw_ixtow.items()}
    else:
        idx2word = {int(v): k for k, v in raw.items()
                    if isinstance(v, (int, str)) and not isinstance(v, dict)}
    print(f"vocab: {vocab_path} size={max(idx2word.keys()) + 1}")

    adapter = get_adapter(args.dataset)
    if args.dataset == 'mtl' and args.root:
        adapter = MTLAdapter(root=args.root)
    print(f"Evaluating {adapter.name} - {len(adapter.keys)} samples, ckpt {args.ckpt} [{args.epoch}]", flush=True)
    res = run_eval(adapter, device, cnn, fc6, score_reg, dive_cls, caption_m, idx2word)

    CLS = ['Position', 'Armstand', 'RotType', 'SSno', 'TWno']
    print(f"\n=== {adapter.name} RESULTS ===")
    print(f"  n={res['n']}  skipped={res['skipped']}")
    print(f"  Spearman rho : {res['rho']:.4f}")
    print(f"  Pearson  r   : {res['pearson']:.4f}")
    print(f"  MSE          : {res['mse']:.2f}")
    print(f"  MAE          : {res['mae']:.2f}")
    print(f"  Classification:")
    for j, name in enumerate(CLS):
        print(f"    {name:<12}: {res['accs'][j]:.2f}%")
    print(f"    {'Average':<12}: {res['avg_acc']:.2f}%")
    out = os.path.join(BASE, f"ssl_{args.dataset}_pred.npy")
    np.save(out, res['pred'])
    np.save(out.replace('pred', 'true'), res['true'])
    print(f"  saved {out}")


if __name__ == '__main__':
    main()