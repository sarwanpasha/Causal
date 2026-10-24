"""Record the Borzoi output layout and the targets table used for the K562 readouts.
Writes readout_config.json in the working directory."""
import json
import os
import sys

import pandas as pd
import torch

sys.path.insert(0, "probes")
from common import device, load_model, path  # noqa: E402

model, _ = load_model("borzoi")
x = torch.zeros(1, 4, 524288, device=device())
x[:, 0, :] = 1.0
with torch.no_grad():
    out = model(x)
d1, d2 = out.shape[1], out.shape[2]
candidates = [path("borzoi_repo", "examples", "targets_human.txt"),
              path("borzoi_repo", "data", "targets_human.txt.gz")]
chosen = None
for c in candidates:
    if os.path.exists(c):
        t = pd.read_csv(c, sep="\t", index_col=0)
        if len(t) in (d1, d2):
            chosen = c
            break
assert chosen, f"no targets file matches output dims {(d1, d2)}"
n_tracks = len(pd.read_csv(chosen, sep="\t", index_col=0))
n_bins = d2 if n_tracks == d1 else d1
json.dump({"n_tracks": n_tracks, "n_bins": n_bins, "targets_file": chosen, "seqlen": 524288},
          open(path("readout_config.json"), "w"), indent=2)
print("tracks", n_tracks, "bins", n_bins, "targets", chosen)
