# convert_c3d.py
# Usage: python convert_c3d.py --pickle c3d.pickle --out C3D_small_PyTorch_Trained_12.pth
import argparse, pickle, torch, torch.nn as nn
from collections import OrderedDict
import numpy as np

class C3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(3, 64, 3, padding=1)
        self.pool1 = nn.MaxPool3d((1,2,2), stride=(1,2,2))
        self.conv2 = nn.Conv3d(64, 128, 3, padding=1)
        self.pool2 = nn.MaxPool3d(2, stride=2)
        self.conv3a = nn.Conv3d(128, 256, 3, padding=1)
        self.conv3b = nn.Conv3d(256, 256, 3, padding=1)
        self.pool3 = nn.MaxPool3d(2, stride=2)
        self.conv4a = nn.Conv3d(256, 512, 3, padding=1)
        self.conv4b = nn.Conv3d(512, 512, 3, padding=1)
        self.pool4 = nn.MaxPool3d(2, stride=2)
        self.conv5a = nn.Conv3d(512, 512, 3, padding=1)
        self.conv5b = nn.Conv3d(512, 512, 3, padding=1)
        self.pool5 = nn.MaxPool3d(2, stride=2)

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pickle", default="c3d.pickle")
    ap.add_argument("--out", default="C3D_small_PyTorch_Trained_12.pth")
    args = ap.parse_args()

    pk = load_pickle(args.pickle)  # dict of numpy arrays
    model = C3D()
    sd = model.state_dict()
    loadable = {}

    # Typical C3D pickle keys look like ".../convXa/weight" or ".../convXa/bias"
    for k, v in pk.items():
        if not isinstance(v, np.ndarray):  # skip non-arrays if any
            continue
        is_w = k.endswith("/weight")
        is_b = k.endswith("/bias")
        if not (is_w or is_b):
            continue

        # Extract 'conv1','conv2','conv3a','conv3b','conv4a','conv4b','conv5a','conv5b'
        parts = k.split("/")
        if len(parts) < 3:
            continue
        lname = parts[-3]  # convXa
        ptype = "weight" if is_w else "bias"
        tgt = f"{lname}.{ptype}"

        # Map Caffe 3D conv weights (T H W C_out? etc) to PyTorch (C_out, C_in, T, H, W) if needed
        arr = v
        if is_w:
            # Heuristic: handle 5D weights (often [T,H,W,C_in,C_out] or [C_out,C_in,T,H,W])
            if arr.ndim == 5 and arr.shape[-1] < 64:  # likely [..., C_out] last
                arr = np.transpose(arr, (4, 3, 0, 1, 2))
            elif arr.ndim == 5 and arr.shape[0] < 64:  # likely [C_in,...,C_out] first
                arr = np.transpose(arr, (4, 0, 1, 2, 3))
        loadable[tgt] = torch.from_numpy(arr).float()

    # Filter by keys present in the model state_dict
    loadable = {k: v for k, v in loadable.items() if k in sd}
    sd.update(loadable)
    model.load_state_dict(sd, strict=False)
    torch.save(model.state_dict(), args.out)
    print(f"Saved: {args.out} with {len(loadable)} layers loaded")

if __name__ == "__main__":
    main()
