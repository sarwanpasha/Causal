"""Shared helpers: paths, sequence encoding, model loading and activation hooks.

All inputs and outputs live under a single working directory given by the
environment variable CAUSAL_WORKDIR (default: ./work). Run
scripts/download_data.sh first to populate it.
"""
import os

import numpy as np

WORKDIR = os.environ.get("CAUSAL_WORKDIR", os.path.join(os.getcwd(), "work"))

# Model geometry (input length, transformer positions, bp per position)
MODEL_SPECS = {
    "borzoi": dict(seqlen=524288, tpos=4096, tres=128, channels_first=True),
    "enformer": dict(seqlen=196608, tpos=1536, tres=128, channels_first=False),
}


def path(*parts):
    return os.path.join(WORKDIR, *parts)


def device():
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(name):
    """Load frozen public weights. Returns (model, number_of_transformer_blocks)."""
    dev = device()
    if name == "borzoi":
        from borzoi_pytorch import Borzoi

        model = Borzoi.from_pretrained("johahi/borzoi-replicate-0")
    elif name == "enformer":
        from enformer_pytorch import from_pretrained

        model = from_pretrained("EleutherAI/enformer-official-rough")
    else:
        raise ValueError(f"unknown model {name}")
    model = model.to(dev).eval()
    return model, len(model.transformer)


def window_sequence(genome, chrom, center, seqlen):
    """Upper-case sequence of length seqlen centred on `center`, N-padded at the ends.

    Returns (sequence, window_start)."""
    half = seqlen // 2
    start = center - half
    seq = str(genome[chrom][max(0, start):start + seqlen]).upper()
    seq = "N" * max(0, -start) + seq
    seq = seq + "N" * (seqlen - len(seq))
    return seq, start


def one_hot(seq, channels_first=True):
    """A/C/G/T one-hot encoding; N and other symbols map to all-zero columns."""
    b = np.frombuffer(seq.encode("ascii"), np.uint8)
    if channels_first:
        a = np.zeros((4, len(b)), np.float32)
        for code, row in zip((65, 67, 71, 84), (0, 1, 2, 3)):
            a[row, b == code] = 1.0
    else:
        a = np.zeros((len(b), 4), np.float32)
        for code, col in zip((65, 67, 71, 84), (0, 1, 2, 3)):
            a[b == code, col] = 1.0
    return a


def shuffle_window(a, s0, s1, rng, channels_first=True):
    """Mononucleotide shuffle of input positions [s0, s1)."""
    a2 = a.copy()
    if channels_first:
        col = a2[:, s0:s1]
        a2[:, s0:s1] = col[:, rng.permutation(col.shape[1])]
    else:
        seg = a2[s0:s1]
        a2[s0:s1] = seg[rng.permutation(len(seg))]
    return a2


def dinucleotide_shuffle_window(a, s0, s1):
    """Dinucleotide-preserving shuffle of positions [s0, s1) (channels-first input)."""
    import torch
    from tangermeme.ersatz import dinucleotide_shuffle

    a2 = a.copy()
    el = torch.from_numpy(a2[:, s0:s1])[None]
    sh = np.asarray(dinucleotide_shuffle(el, n=1)).reshape(-1, 4, s1 - s0)[0]
    a2[:, s0:s1] = sh.astype(np.float32)
    return a2


def forward(model, name, a):
    """Run one forward pass on a single encoded sequence and return the output tensor."""
    import torch

    x = torch.from_numpy(a)[None].to(device())
    if name == "enformer":
        try:
            out = model(x, head="human")
        except TypeError:
            out = model(x)
        if isinstance(out, dict):
            out = out["human"]
        return out
    return model(x)


class PairCapture:
    """Forward hook that stores the block output at two transformer positions."""

    def __init__(self):
        self.pA = self.pB = None
        self.vA = self.vB = None

    def __call__(self, module, inputs, output):
        t = output[0] if isinstance(output, (tuple, list)) else output
        if self.pA is not None:
            self.vA = t[0, self.pA, :].detach().clone()
            self.vB = t[0, self.pB, :].detach().clone()
        return output


class FullCapture:
    """Forward hook that stores the whole block output (positions x channels)."""

    def __init__(self):
        self.v = None

    def __call__(self, module, inputs, output):
        t = output[0] if isinstance(output, (tuple, list)) else output
        self.v = t[0].detach()
        return output


class Patch:
    """Forward hook used for interchange interventions on a set of positions.

    mode == "cache": store the block output at `pos` (run on the counterfactual input)
    mode == "patch": overwrite the block output at `pos` with the stored activations
    mode == "off":   do nothing
    """

    def __init__(self):
        self.mode = "off"
        self.pos = None
        self.store = None

    def __call__(self, module, inputs, output):
        t = output[0] if isinstance(output, (tuple, list)) else output
        if self.mode == "cache":
            self.store = t[:, self.pos, :].detach().clone()
        elif self.mode == "patch" and self.store is not None:
            t[:, self.pos, :] = self.store
        return output


def pair_similarity(vA, vB):
    """Similarity descriptor of Eq. (1): cosine and normalized distance."""
    import torch

    cos = float(torch.nn.functional.cosine_similarity(vA, vB, dim=0))
    nd = float(torch.linalg.norm(vA - vB) / (torch.linalg.norm(vA) + torch.linalg.norm(vB) + 1e-6))
    return cos, nd


def to_bool(series):
    return series.astype(str).str.upper().isin(["TRUE", "1", "T"])
