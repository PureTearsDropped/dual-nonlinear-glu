"""Benchmark for dual-nonlinear-glu.

Seeds are stacked into a leading dimension, so `--seeds 8` costs little more
than `--seeds 1`. Gradient clipping is per-seed; a global norm would couple the
models.

    python bench.py --diagnose
    python bench.py --depth 32 --seeds 8 --model fg,swiglu,gelu
    python bench.py --depth 16 --seeds 8 --model fg,ff,gg --teacher deep

0BSD.
"""
import argparse, math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DIN, DOUT = 32, 8


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


pos = lambda r: F.softplus(r) + 1e-4
inv = lambda v: math.log(math.expm1(max(v - 1e-4, 1e-6)))
f_ab = lambda x, a, b: torch.where(x >= 0, a * x, b * torch.tanh(a * x / b))
g_AB = lambda y, A, B: torch.where(y >= 0, y / A, (B / A) * torch.asinh(y / B))

# Cheap replacements. f becomes a single max; g becomes one rsqrt.
#   tanh(u) -> clamp(u,-1,1)  makes f_ab(x) = max(a x, -b) exactly
#   asinh(u) -> u / sqrt(1 + u^2/3)
# Branch-free forms. These kernels are bandwidth-bound, not ALU-bound: on GPU
# `torch.asinh` costs about as much as an add. A `where` reads three tensors
# and a condition where `asinh` reads one, and measures 1.2-2.4x an `asinh`
# depending on size (RTX 5090, f32, N = 65k .. 32M). Dropping a branch saves
# tensor traffic, not arithmetic.
#   f: tanh -> clamp makes the two branches collapse into one max
#   g: apply asinh on both sides, dropping the branch (this changes the
#      function -- the positive side is compressed too)
f_hard = lambda x, a, b: torch.maximum(a * x, -b)
g_alg  = lambda y, A, B: (B / A) * torch.asinh(y / B)

# Piecewise asinh -- polynomials only, for hardware with no transcendental
# unit. Imported rather than copied so the coefficients have one home.
# These are the *same* functions as g_alg / g_AB to within 1e-7, not cheaper
# variants: on a GPU they are ~30x slower. See APPROX.md.
from dnglu import asinh_pw
g_pw   = lambda y, A, B: (B / A) * asinh_pw(y / B)                       # = g_alg
g_ABpw = lambda y, A, B: torch.where(y >= 0, y / A,
                                     (B / A) * asinh_pw(y / B))          # = g_AB


class Teacher(nn.Module):
    """kind: tanh / relu / gelu products, or `deep` (an iterated map, no product)."""
    def __init__(self, width=128, kind="tanh"):
        super().__init__()
        self.kind = kind
        self.A = nn.Parameter(torch.randn(width, DIN) * DIN ** -0.5)
        self.B = nn.Parameter(torch.randn(width, DIN) * DIN ** -0.5)
        self.o = nn.Linear(width, DOUT)

    def forward(self, x):
        p, q = x @ self.A.t(), x @ self.B.t()
        if self.kind == "tanh":   z = torch.tanh(p) * torch.tanh(q)
        elif self.kind == "relu": z = F.relu(p) * F.relu(q)
        elif self.kind == "gelu": z = F.gelu(p) * F.gelu(q)
        elif self.kind == "deep":
            z = p
            for _ in range(3): z = torch.tanh(2.0 * z) + 0.5 * q
        else: raise ValueError(self.kind)
        return self.o(z)


def _gens(S): return [torch.Generator().manual_seed(1000 + i) for i in range(S)]
def _u(shape, fan, g): return (torch.rand(shape, generator=g) * 2 - 1) * (fan ** -0.5)


class _Stack(nn.Module):
    def __init__(self, S, W, gen):
        super().__init__()
        self.Wi = nn.Parameter(torch.stack([_u((W, DIN), DIN, g) for g in gen]))
        self.bi = nn.Parameter(torch.stack([_u((W,), DIN, g) for g in gen]))
        self.Wo = nn.Parameter(torch.stack([_u((DOUT, W), W, g) for g in gen]))
        self.bo = nn.Parameter(torch.stack([_u((DOUT,), W, g) for g in gen]))
    def head(self, x): return torch.einsum('bi,sni->sbn', x, self.Wi) + self.bi[:, None, :]
    def tail(self, h): return torch.einsum('sbn,son->sbo', h, self.Wo) + self.bo[:, None, :]


class Gated(_Stack):
    """order: two letters from {f, g, s(ilu), i(dentity)} -- gate, then value.

    fg = this work.  si = SwiGLU.  fi / sg isolate the two branches.
    """
    def __init__(self, S, W=64, depth=4, order="fg", scaled=True, m=None):
        gen = _gens(S); super().__init__(S, W, gen)
        self.d, self.order = depth, order
        M = m or max(1, W // 3); self.m = M
        ia, ib = inv(1.0), inv(0.1)
        sd = M ** -0.5 * (depth ** -0.5 if scaled else 1.0)
        st = lambda sh, s: nn.Parameter(torch.stack(
            [torch.randn(sh, generator=g) * s for g in gen]))
        self.Wg = st((depth, M, W), W ** -0.5)
        self.Wu = st((depth, M, W), W ** -0.5)
        self.Wd = st((depth, W, M), sd)
        self.bm = nn.Parameter(torch.zeros(S, depth, W))
        mk = lambda v: nn.Parameter(torch.full((S, depth, M), v))
        self.a, self.b, self.A, self.B = mk(ia), mk(ib), mk(ia), mk(ib)

    def _br(self, t, which, a, b, A, B):
        return {"f": lambda: f_ab(t, a, b), "g": lambda: g_AB(t, A, B),
                "F": lambda: f_hard(t, a, b), "G": lambda: g_alg(t, A, B),
                "p": lambda: g_pw(t, A, B),  "P": lambda: g_ABpw(t, A, B),
                "s": lambda: F.silu(t),     "i": lambda: t,
                "r": lambda: F.relu(t),     "e": lambda: F.gelu(t)}[which]()

    def forward(self, x):
        h = self.head(x)
        for L in range(self.d):
            a, b = pos(self.a[:, L])[:, None, :], pos(self.b[:, L])[:, None, :]
            A, B = pos(self.A[:, L])[:, None, :], pos(self.B[:, L])[:, None, :]
            gt = torch.bmm(h, self.Wg[:, L].transpose(1, 2))
            up = torch.bmm(h, self.Wu[:, L].transpose(1, 2))
            z = self._br(gt, self.order[0], a, b, A, B) * self._br(up, self.order[1], a, b, A, B)
            h = h + torch.bmm(z, self.Wd[:, L].transpose(1, 2)) + self.bm[:, L][:, None, :]
        return self.tail(h)


class MLP(_Stack):
    def __init__(self, S, W=64, depth=4, act="gelu", scaled=True):
        gen = _gens(S); super().__init__(S, W, gen)
        self.d, self.act = depth, act
        sc = W ** -0.5 * (depth ** -0.5 if scaled else 1.0)
        self.Wm = nn.Parameter(torch.stack(
            [torch.randn(depth, W, W, generator=g) * sc for g in gen]))
        self.bm = nn.Parameter(torch.zeros(S, depth, W))
    def forward(self, x):
        h = self.head(x); act = {"gelu": F.gelu, "relu": F.relu, "silu": F.silu}[self.act]
        for L in range(self.d):
            h = h + torch.bmm(act(h), self.Wm[:, L].transpose(1, 2)) + self.bm[:, L][:, None, :]
        return self.tail(h)


class Linear(_Stack):
    def __init__(self, S, W=64, **kw): super().__init__(S, W, _gens(S))
    def forward(self, x): return self.tail(self.head(x))


class OneNonlin(_Stack):
    def __init__(self, S, W=64, **kw):
        super().__init__(S, W, _gens(S))
        self.a = nn.Parameter(torch.full((S, W), inv(1.0)))
        self.b = nn.Parameter(torch.full((S, W), inv(0.1)))
    def forward(self, x):
        return self.tail(f_ab(self.head(x), pos(self.a)[:, None, :], pos(self.b)[:, None, :]))


class Ablate(Gated):
    """Interior blocks zeroed: only the input and output projections act."""
    def forward(self, x): return self.tail(self.head(x))


MODELS = {
    "fg": lambda S, **k: Gated(S, order="fg", **k),      # this work
    "FG": lambda S, **k: Gated(S, order="FG", **k),      # cheap: max() and one rsqrt
    "Fg": lambda S, **k: Gated(S, order="Fg", **k),      # cheap gate only
    "fG": lambda S, **k: Gated(S, order="fG", **k),      # cheap value only
    "gf": lambda S, **k: Gated(S, order="gf", **k),
    "ff": lambda S, **k: Gated(S, order="ff", **k),
    "gg": lambda S, **k: Gated(S, order="gg", **k),
    "sg": lambda S, **k: Gated(S, order="sg", **k),      # silu gate, g value
    "fi": lambda S, **k: Gated(S, order="fi", **k),      # f gate, identity value
    "swiglu": lambda S, **k: Gated(S, order="si", **k),  # silu gate, identity value
    "bilinear": lambda S, **k: Gated(S, order="ii", **k),  # Shazeer 2020: both linear
    "geglu": lambda S, **k: Gated(S, order="ei", **k),
    "reglu": lambda S, **k: Gated(S, order="ri", **k),
    "ss": lambda S, **k: Gated(S, order="ss", **k),      # silu on both branches
    "gelu": lambda S, **k: MLP(S, act="gelu", **k),
    "relu": lambda S, **k: MLP(S, act="relu", **k),
    "silu": lambda S, **k: MLP(S, act="silu", **k),
    "linear": lambda S, **k: Linear(S, **k),
    "onenl": lambda S, **k: OneNonlin(S, **k),
    "ablate": lambda S, **k: Ablate(S, order="fg", **k),
}


def clip_per_seed(model, S, maxnorm=5.0):
    for p in model.parameters():
        if p.grad is None: continue
        g = p.grad.reshape(S, -1)
        g.mul_((maxnorm / (g.norm(dim=1, keepdim=True) + 1e-6)).clamp(max=1.0))


def run(name, S, depth, steps, width, scaled, teacher, dev, swap_to=None):
    set_seed(1000); T = Teacher(kind=teacher).to(dev).eval()
    for p in T.parameters(): p.requires_grad_(False)
    set_seed(0)
    kw = dict(W=width, depth=depth, scaled=scaled)
    if name in ("linear", "onenl"): kw = dict(W=width)
    m = MODELS[name](S, **kw).to(dev)
    npar = sum(p.numel() for p in m.parameters()) // S
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        x = torch.randn(512, DIN, device=dev)
        with torch.no_grad(): y = T(x)
        opt.zero_grad(set_to_none=True)
        ((m(x) - y[None]) ** 2).mean(dim=(1, 2)).sum().backward()
        clip_per_seed(m, S); opt.step(); sch.step()
    with torch.no_grad():
        x = torch.randn(20000, DIN, device=dev); y = T(x)
        loss = ((m(x) - y[None]) ** 2).mean(dim=(1, 2)) / y.var()
        if swap_to is not None:            # same weights, asinh -> polynomials
            m.order = swap_to
            loss2 = ((m(x) - y[None]) ** 2).mean(dim=(1, 2)) / y.var()
            return [float(v) for v in loss], npar, [float(v) for v in loss2]
    return [float(v) for v in loss], npar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="fg,swiglu,gelu")
    ap.add_argument("--teacher", default="tanh", choices=["tanh", "relu", "gelu", "deep"])
    ap.add_argument("--depth", type=int, default=32)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--no-scale", action="store_true",
                    help="drop the 1/sqrt(L) init on W_d (SPEC 5)")
    ap.add_argument("--swap", action="store_true",
                    help="after training, re-evaluate the same weights with "
                         "asinh replaced by polynomials (same function, "
                         "see APPROX.md)")
    ap.add_argument("--diagnose", action="store_true",
                    help="baselines that decide whether the task is usable")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    names = ["linear", "onenl", "gelu", "ablate"] if a.diagnose else a.model.split(",")
    print(f"teacher {a.teacher}  depth {a.depth}  width {a.width}  {a.steps} steps  "
          f"{a.seeds} seeds  W_d init {'1/sqrt(L)' if not a.no_scale else 'plain'}  [{dev}]")
    print(f"{'model':>10} {'loss':>10} {'std':>9} {'params':>10}")
    print("-" * 44)
    res = {}
    # Function-preserving swaps: the transcendental is replaced by polynomials,
    # the function is not changed. fg -> FG would be a different function (the
    # value branch's positive half gets compressed), which is what RESULTS 4
    # measures, not what --swap is for.
    SWAP = {"FG": "Fp", "fG": "fp", "fg": "fP", "Fg": "FP"}
    for n in names:
        d = 1 if n in ("linear", "onenl") else a.depth
        sw = SWAP.get(n) if a.swap else None
        out = run(n, a.seeds, d, a.steps, a.width, not a.no_scale, a.teacher, dev, sw)
        L, npar = out[0], out[1]
        res[n] = (np.mean(L), np.std(L, ddof=1))
        line = f"{n:>10} {np.mean(L):10.5f} {np.std(L, ddof=1):9.5f} {npar:10,}"
        if len(out) > 2:
            L2 = out[2]
            line += f"   swapped: {np.mean(L2):.5f} ({(np.mean(L2)-np.mean(L))/np.mean(L)*100:+.1f}%)"
        print(line)
    if a.diagnose:
        v = res["onenl"][0] / res["gelu"][0]
        print(f"\ndepth value = {v:.2f}  "
              f"{'OK' if v >= 1.5 else 'TOO LOW - this task does not need depth'}")
        print(f"ablation {res['ablate'][0]:.5f} must be clearly worse than any real model")
    elif len(names) > 1:
        base = names[0]
        print(f"\ndifferences against {base}, in standard errors ({a.seeds} seeds):")
        m0, s0 = res[base]
        for n in names[1:]:
            m1, s1 = res[n]
            se = math.sqrt(s0 ** 2 / a.seeds + s1 ** 2 / a.seeds)
            print(f"  {n:>10}  {(m1 - m0) / m0 * 100:+7.1f}%   {(m1 - m0) / max(se, 1e-12):6.1f} sigma")


if __name__ == "__main__":
    main()
